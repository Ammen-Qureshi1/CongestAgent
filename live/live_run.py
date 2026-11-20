"""
ONE live (closed-loop) run: the same emulated network and traffic as the dataset runs, but a controller
reads measurements every 250 ms and may REROUTE source-destination pairs.

Policies (identical rerouting logic; only the congestion DETECTOR differs):
  none       shortest-path routing, never reroutes (B1)
  delay      a link is congested if delay > propagation + 40 ms   (best rule tuned on training data)
  delayloss  delay > +40 ms OR path loss > 5%                      (best tuned delay-or-loss rule)
  delayutil  delay > +40 ms AND busy link (window utilisation >= the gate)   (fair baseline: same gate as the model)
  model      CongestAgent perception model (LightGBM): congested if it predicts 'congestion' on a busy link (gate)

Usage (as root):  python3 live_run.py nsfnet 90 shift80 mixed none 11 model
Output: live_runs/<run_id>__<policy>/ (raw data, controller log, intervals + labels)
"""
import os, sys, json, time, glob, shutil, threading, subprocess, resource
from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'datagen')); sys.path.insert(0, str(HERE))
from common import (RUN_S, PROBE_HZ, PROBE_PORT, TCP_PORT, UDP_PORT, SANDBOX, USERSPACE_OVS, INTERVAL_S, WARMUP_S,
                    CAPACITY_MBPS, QUEUE_LIMIT_BYTES, PATHLOSS_WINDOW_S, load_topology, host_ip, write_json)
from netbuild import Network
from scenario import make_schedule, run_id
import collector as _collector
from collector import Collector
_collector.FLOWSTAT_S = 0.5      # live runs poll OpenFlow flow counters every 0.5 s (dataset: 1 s) so path loss is fresher
from features_live import FeatureEngine, F_COLS

LIVE_DIR = Path(os.environ.get('CA_LIVE_DIR', HERE.parent / 'live_runs'))
MODEL = Path(os.environ.get('CA_LIVE_MODEL', HERE.parent / 'models' / 'results' / 'lgbm_live.txt'))
START_DELAY_S = 4.0
LAG_S = 0.35            # the controller judges interval k at its end + 0.35 s (late probes have arrived)
HOLD_DIR_S = 2.0        # after acting on a link, wait 2 s before acting on it again
HOLD_PAIR_S = 5.0       # a pair is not moved again within 5 s
REVERT_S = 10.0         # a moved pair returns to its shortest path after its old path is clean for 10 s
MAX_MOVES = 3           # at most 3 pairs moved per 250 ms step
DELAY_RULE_MS, LOSS_RULE = 40.0, 0.05      # defaults; replaced by the thresholds re-tuned on training data, if present
_RULES = HERE.parent / 'models' / 'results' / 'live_rules.json'
if _RULES.exists():
    _r = json.loads(_RULES.read_text()); DELAY_RULE_MS, LOSS_RULE = float(_r['delay_ms']), float(_r['loss'])
    DELAYLOSS_DELAY_MS = float(_r.get('delayloss_delay_ms', DELAY_RULE_MS))
    UTIL_GATE = float(_r.get('util_gate', 0.0))
else:
    DELAYLOSS_DELAY_MS = DELAY_RULE_MS
    UTIL_GATE = 0.0       # 0 = no gate
MIN_PKTS = 20
NS = 1_000_000_000
POLICIES = ('none', 'delay', 'delayloss', 'delayutil', 'model', 'model_robust')
# model_robust (added after the first RQ1 results): the 'robust' LightGBM of CongestAgent/train_perception_colab.py (no raw delay-level features,
# its own 5 s / 10 s rolling features) + the same busy-link gate (CA_ROBUST_GATE=0 switches the gate off)
ROBUST_DIR = Path(os.environ.get('CA_ROBUST_DIR', HERE.parent / 'models' / 'results' / 'robust'))
ROBUST_GATE = os.environ.get('CA_ROBUST_GATE', '1') == '1'
# RQ2: same perception model, different control architecture
RQ2_POLICIES = ('central', 'regional', 'regional_nofast', 'regional_v2')
# regional_v2 (revised after the first RQ2 results): a region triggers a reroute from another region's summary only
# if that link was flagged in at least half of the last summary window (persistent); its own links are always fresh.
SLOW_EVERY = 8         # regional agents: slow loop every 8 intervals (2 s); 'regional_nofast' runs it every interval


def ofctl(*args):
    return subprocess.run(['ovs-ofctl', '-O', 'OpenFlow13', *args], capture_output=True, text=True)


class Controller(threading.Thread):
    def __init__(self, net, topo, policy, raw, t0, col):
        super().__init__(daemon=True)
        self.net, self.policy, self.raw, self.t0, self.col = net, policy, Path(raw), t0, col
        self.n_nodes = len(topo['nodes'])
        self.dirs = sorted(net.dirs, key=lambda d: d['dir_id'])
        self.nd = len(self.dirs)
        self.dir_of = {(d['src'], d['dst']): d['dir_id'] for d in self.dirs}
        lk = {l['id']: l for l in topo['links']}
        prop = [lk[d['link_id']]['delay_ms'] for d in self.dirs]
        self.prop = np.array(prop, float)
        self.eng = FeatureEngine([(d['src'], d['dst']) for d in self.dirs], prop, CAPACITY_MBPS, QUEUE_LIMIT_BYTES)
        self.G = nx.DiGraph()
        for d in self.dirs:
            self.G.add_edge(d['src'], d['dst'], dir=d['dir_id'])
        self.orig = {p: list(v) for p, v in net.paths.items()}
        self.paths = {p: list(v) for p, v in net.paths.items()}
        self.has_rule = {p: set(v) for p, v in net.paths.items()}
        self.sw = {i: s.name for i, s in net.switches.items()}
        self.model = None
        if policy == 'model' or policy in RQ2_POLICIES:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(MODEL))
            self.mcols = self.model.feature_name()
        if policy == 'model_robust':
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(ROBUST_DIR / 'model.txt'))
            self.mcols = json.loads((ROBUST_DIR / 'features.json').read_text())
        self.K = int(round(RUN_S / INTERVAL_S))
        self.flags = np.zeros((self.K, self.nd), np.int8)          # detector output per interval and direction
        self.pred = np.full((self.K, self.nd), -1, np.int8)         # model class (model policy only)
        self.fhist = {c: np.full((self.K, self.nd), np.nan, np.float32) for c in F_COLS}   # what the controller saw
        self.steps, self.moves = [], []
        self.last_dir_act = {}; self.last_pair_change = {}; self.last_flag_t = np.full(self.nd, -1e9)
        self.n_flowmods = 0
        # live inputs
        self.port_i = 0; self.ports = {}
        self.flow_i = 0; self.fl = {}
        self.fh = {}; self.buf = {}
        self.sends = {}; self.rcv = {}; self.maxseq = {}
        self.errors = []
        self.fm_remote = 0; self.msgs_longhaul = 0; self.msgs_local = 0
        self.det_ms_r = []; self.act_ms_r = []; self.ctl_rtt_ms = 0.0; self.n_reg = 1
        if policy in RQ2_POLICIES:
            self._init_rq2(topo)

    def _init_rq2(self, topo):
        """Regions, three candidate paths per aggregate, and the emulated control-channel round trip."""
        self.region_of = {n['id']: n.get('region', 0) for n in topo['nodes']}
        self.n_reg = max(self.region_of.values()) + 1
        self.own = [np.array([self.region_of[d['src']] == r for d in self.dirs]) for r in range(self.n_reg)]
        self.dir_idx = [np.flatnonzero(m) for m in self.own]
        self.n_summary = len({(self.region_of[l['a']], self.region_of[l['b']]) for l in topo['links'] if self.region_of[l['a']] != self.region_of[l['b']]} |
                             {(self.region_of[l['b']], self.region_of[l['a']]) for l in topo['links'] if self.region_of[l['a']] != self.region_of[l['b']]})
        self.slow_every = 1 if self.policy == 'regional_nofast' else SLOW_EVERY
        self.snap = np.zeros(self.nd, bool); self.snap_pers = np.zeros(self.nd, bool)
        Gd = nx.Graph()
        for l in topo['links']:
            Gd.add_edge(l['a'], l['b'], w=l['delay_ms'])
        self.cand = {}
        for p in self.paths:
            others = []
            for c in nx.shortest_simple_paths(Gd, p[0], p[1], weight='w'):
                if list(c) != self.orig[p]:
                    others.append(list(c))
                if len(others) == 2:
                    break
            self.cand[p] = [list(self.orig[p])] + others
        dist = dict(nx.all_pairs_dijkstra_path_length(Gd, weight='w'))
        if self.policy == 'central':          # one agent at the most central node, reports from every switch
            self.ctl_rtt_ms = 2 * min(max(dist[a].values()) for a in Gd.nodes)
        else:                                 # one agent per region, only its own switches
            rt = []
            for r in range(self.n_reg):
                nodes = [n for n in Gd.nodes if self.region_of[n] == r]
                rt.append(2 * min(max(dist[a][n] for n in nodes) for a in nodes))
            self.ctl_rtt_ms = max(rt)
        self.ctl_rtt_s = self.ctl_rtt_ms / 1000.0

    # ---------------- reading live measurements ----------------
    def _tail(self, path):
        if path not in self.fh:
            if not os.path.exists(path):
                return []
            self.fh[path] = open(path); self.buf[path] = ''
        data = self.buf[path] + self.fh[path].read()
        lines = data.split('\n'); self.buf[path] = lines[-1]
        return [x for x in lines[:-1] if x]

    def _ingest(self):
        rows = self.col.port_rows; n = len(rows)
        for r in rows[self.port_i:n]:
            self.ports.setdefault(r[1], []).append(r)
        self.port_i = n
        rows = self.col.flow_rows; n = len(rows)
        for r in rows[self.flow_i:n]:
            self.fl.setdefault((r[1], r[2]), []).append((r[0], r[3]))
        self.flow_i = n
        for i in range(self.n_nodes):
            for line in self._tail(str(self.raw / f'live_tx_{i}.csv')):
                dst, seq, s = line.split(','); s = int(s)
                self.sends.setdefault((s - self.t0_ns) // int(INTERVAL_S * NS), []).append((i, int(dst), int(seq), s))
            for line in self._tail(str(self.raw / f'live_rx_{i}.csv')):
                src, seq, s, r = line.split(','); src, seq = int(src), int(seq)
                m = self.maxseq.get((src, i), -1)
                self.rcv[(src, i, seq)] = (int(r), seq < m)
                self.maxseq[(src, i)] = max(m, seq)

    @staticmethod
    def _diff(series, idx, a, b):
        if not series or len(series) < 2:
            return np.nan
        ts = np.array([x[0] for x in series[-12:]], float); v = np.array([x[idx] for x in series[-12:]], float)
        c = np.interp([a, b], ts, v, left=np.nan, right=np.nan)
        return c[1] - c[0]

    def _interval(self, k):
        a = self.t0_ns + k * int(INTERVAL_S * NS); b = a + int(INTERVAL_S * NS)
        f = {c: np.full(self.nd, np.nan) for c in F_COLS}
        for d in self.dirs:
            i = d['dir_id']; e, g = self.ports.get(d['egress_intf']), self.ports.get(d['ingress_intf'])
            txb, txp, txd = self._diff(e, 2, a, b), self._diff(e, 3, a, b), self._diff(e, 6, a, b)
            rxp, rxd = self._diff(g, 5, a, b), self._diff(g, 7, a, b)
            f['f_util'][i] = txb * 8 / (INTERVAL_S * CAPACITY_MBPS * 1e6); f['f_tx_mbps'][i] = txb * 8 / INTERVAL_S / 1e6
            f['f_tx_pps'][i] = txp / INTERVAL_S; f['f_rx_pps_far'][i] = rxp / INTERVAL_S
            f['f_tx_dropped'][i] = txd; f['f_rx_dropped_far'][i] = rxd
        # probes sent in [a, b)
        per = {}
        for (u, v, seq, s) in self.sends.pop(k, []):
            per.setdefault((u, v), []).append((seq, s))
        for kk in [x for x in self.sends if x < k]:
            del self.sends[kk]
        for (u, v), lst in per.items():
            i = self.dir_of.get((u, v))
            if i is None:
                continue
            dl, ro = [], 0
            for seq, s in lst:
                r = self.rcv.get((u, v, seq))
                if r:
                    dl.append((r[0] - s) / 1e6); ro += int(r[1])
            f['f_probe_sent'][i] = len(lst); f['f_probe_recv'][i] = len(dl)
            if dl:
                x = np.array(dl)
                f['f_delay_mean_ms'][i], f['f_delay_min_ms'][i], f['f_delay_max_ms'][i] = x.mean(), x.min(), x.max()
                f['f_delay_std_ms'][i], f['f_delay_p50_ms'][i] = x.std(), np.median(x)
                f['f_probe_reordered'][i] = ro
        for c in ('f_probe_sent', 'f_probe_recv'):
            f[c] = np.nan_to_num(f[c])
        f['f_probe_lost'] = f['f_probe_sent'] - f['f_probe_recv']
        with np.errstate(invalid='ignore', divide='ignore'):
            f['f_probe_loss_ratio'] = np.where(f['f_probe_sent'] > 0, f['f_probe_lost'] / f['f_probe_sent'], np.nan)
        # path loss over the last 2 s (OpenFlow flow counters of the first and last switch of each pair)
        self.pair_pk = {}
        L = {i: [] for i in range(self.nd)}
        for (s_, t_), path in self.paths.items():
            c = s_ * self.n_nodes + t_ + 1
            f1, f2 = self.fl.get((self.sw[s_], c)), self.fl.get((self.sw[t_], c))
            if not f1 or not f2 or len(f1) < 2 or len(f2) < 2:
                continue
            end = min(b, f1[-1][0], f2[-1][0]); st = end - int(PATHLOSS_WINDOW_S * NS)
            first = self._diff(f1, 1, st, end); last = self._diff(f2, 1, st, end)
            if not first >= MIN_PKTS:
                continue
            loss = float(np.clip((first - last) / first, 0, 1))
            self.pair_pk[(s_, t_)] = first
            for u, v in zip(path[:-1], path[1:]):
                L[self.dir_of[(u, v)]].append((loss, first))
        f['f_pathloss_wmean'] = np.full(self.nd, np.nan); f['f_pathloss_max'] = np.full(self.nd, np.nan)
        f['f_path_pairs_active'] = np.zeros(self.nd); f['f_path_pkts_2s'] = np.zeros(self.nd)
        for i, lst in L.items():
            if lst:
                lo, w = np.array(lst).T
                f['f_pathloss_wmean'][i] = (lo * w).sum() / w.sum(); f['f_pathloss_max'][i] = lo.max()
                f['f_path_pairs_active'][i] = len(lst); f['f_path_pkts_2s'][i] = w.sum()
        return f

    # ---------------- detection ----------------
    def _detect(self, k, f, feats):
        if self.policy == 'none':
            return np.zeros(self.nd, bool)
        excess = f['f_delay_mean_ms'] - self.prop
        with np.errstate(invalid='ignore'):
            if self.policy == 'delay':
                return excess > DELAY_RULE_MS
            if self.policy == 'delayloss':
                return (excess > DELAYLOSS_DELAY_MS) | (f['f_pathloss_wmean'] > LOSS_RULE)
            if self.policy == 'delayutil':       # delay rule + the same utilisation gate as the model (fair baseline)
                return (excess > DELAY_RULE_MS) & (feats['w_util_mean'] >= UTIL_GATE)
        if self.policy == 'model_robust':
            X = np.column_stack([self._robust_col(c, k, f, feats) for c in self.mcols])
            cls = self.model.predict(X).argmax(1)
            self.pred[k] = cls
            ok = (cls == 2) & feats['w_complete'].astype(bool)
            if ROBUST_GATE:
                with np.errstate(invalid='ignore'):
                    ok &= feats['w_util_mean'] >= UTIL_GATE
            return ok
        if self.policy in ('regional', 'regional_nofast', 'regional_v2'):
            cls = np.full(self.nd, -1); self.det_ms_r = []
            for r in range(self.n_reg):
                t_ = time.perf_counter(); ix = self.dir_idx[r]
                cls[ix] = self.model.predict(np.column_stack([feats[c][ix] for c in self.mcols])).argmax(1)
                self.det_ms_r.append((time.perf_counter() - t_) * 1000)
        else:
            X = np.column_stack([feats[c] for c in self.mcols])
            cls = self.model.predict(X).argmax(1)
        self.pred[k] = cls
        with np.errstate(invalid='ignore'):
            busy = feats['w_util_mean'] >= UTIL_GATE      # 'congestion' only counts on a busy link
        return (cls == 2) & feats['w_complete'].astype(bool) & busy

    def _robust_col(self, c, k, f, feats):
        """Inputs of the robust model, built exactly like train_perception_colab.py: f_util clipped at 1,
        long_<x>_5s / _10s = mean over the last 20 / 40 intervals (fewer at the start), NaN ignored."""
        if not c.startswith('long_'):
            v = np.asarray(feats[c] if c in feats else f[c], float)
            return np.minimum(v, 1.0) if c == 'f_util' else v
        name, sec = c[5:].rsplit('_', 1); n = 20 if sec == '5s' else 40
        lo = max(0, k - n + 1)
        H = {x: np.vstack([self.fhist[x][lo:k], np.asarray(f[x], float)[None]]) for x in
             ('f_tx_pps', 'f_rx_pps_far', 'f_pathloss_wmean', 'f_delay_std_ms', 'f_probe_loss_ratio', 'f_util',
              'f_delay_max_ms', 'f_delay_min_ms', 'f_delay_mean_ms')}
        if name == 'pathloss': v = H['f_pathloss_wmean']
        elif name == 'portloss':
            tx = np.where(H['f_tx_pps'] == 0, np.nan, H['f_tx_pps']); v = np.clip((H['f_tx_pps'] - H['f_rx_pps_far']) / tx, 0, 1)
        elif name == 'delay_spread': v = H['f_delay_std_ms']
        elif name == 'probe_loss': v = H['f_probe_loss_ratio']
        elif name == 'util': v = np.minimum(H['f_util'], 1.0)
        elif name == 'delay_range': v = H['f_delay_max_ms'] - H['f_delay_min_ms']
        elif name == 'excess_delay': v = H['f_delay_mean_ms'] - self.prop[None]
        else: raise KeyError(c)
        with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore')
            return np.nanmean(v.astype(float), axis=0)

    # ---------------- acting ----------------
    def _install(self, pair, path, region=None):
        a, b = pair; c = a * self.n_nodes + b + 1
        for idx in range(len(path) - 1, -1, -1):          # destination side first
            u = path[idx]
            out = self.net.host_port[u] if u == b else self.net.iface[(u, path[idx + 1])]['port']
            rule = f'table=0,priority=100,ip,nw_src={host_ip(a)},nw_dst={host_ip(b)},actions=output:{out}'
            r = ofctl('--strict', 'mod-flows', self.sw[u], rule) if u in self.has_rule[pair] else \
                ofctl('add-flow', self.sw[u], f'cookie={c},{rule}')
            self.n_flowmods += 1
            if region is None or self.region_of.get(u, -1) != region:
                self.fm_remote += 1
            if r.returncode != 0:
                self.errors.append(f'flowmod {self.sw[u]} {pair}: {r.stderr.strip()[:120]}')
            self.has_rule[pair].add(u)
        self.paths[pair] = list(path)

    def _act(self, k, now_s, flagged, f):
        moves = []
        bad = set(np.flatnonzero(flagged).tolist())
        self.last_flag_t[list(bad)] = now_s
        H = self.G.copy()
        H.remove_edges_from([(self.dirs[i]['src'], self.dirs[i]['dst']) for i in bad])
        order = sorted(bad, key=lambda i: -np.nan_to_num(f['f_util'][i]))
        for i in order:
            if len(moves) >= MAX_MOVES:
                break
            if now_s - self.last_dir_act.get(i, -1e9) < HOLD_DIR_S:
                continue
            u, v = self.dirs[i]['src'], self.dirs[i]['dst']
            cand = [p for p, pth in self.paths.items()
                    if any((x, y) == (u, v) for x, y in zip(pth[:-1], pth[1:]))
                    and now_s - self.last_pair_change.get(p, -1e9) >= HOLD_PAIR_S and self.pair_pk.get(p, 0) > 0]
            if not cand:
                continue
            p = max(cand, key=lambda q: self.pair_pk.get(q, 0))
            try:
                new = nx.shortest_path(H, p[0], p[1])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if new == self.paths[p]:
                continue
            old = self.paths[p]; t = time.perf_counter(); self._install(p, new)
            moves.append(dict(kind='reroute', k=k, t_s=round(now_s, 3), trigger_dir=i, pair=list(p), old=old, new=new,
                              install_ms=round((time.perf_counter() - t) * 1000, 2)))
            self.last_dir_act[i] = now_s; self.last_pair_change[p] = now_s
        for p in list(self.paths):
            if self.paths[p] == self.orig[p] or now_s - self.last_pair_change.get(p, -1e9) < REVERT_S:
                continue
            od = [self.dir_of[(x, y)] for x, y in zip(self.orig[p][:-1], self.orig[p][1:])]
            if all(now_s - self.last_flag_t[j] >= REVERT_S for j in od):
                old = self.paths[p]; t = time.perf_counter(); self._install(p, self.orig[p])
                moves.append(dict(kind='revert', k=k, t_s=round(now_s, 3), trigger_dir=-1, pair=list(p), old=old,
                                  new=self.orig[p], install_ms=round((time.perf_counter() - t) * 1000, 2)))
                self.last_pair_change[p] = now_s
        return moves

    def _act_regional(self, k, now_s, flagged, f):
        """Regional agents. Every agent sees its own links fresh and other regions' links only through the last summary.
        Fast loop (every interval): move an aggregate to its best clean backup (of 3 candidates) when a link on its
        primary path is flagged. Slow loop (every SLOW_EVERY intervals): exchange summaries and return aggregates
        to their primary path when it has been clean for REVERT_S. 'regional_nofast' runs the slow loop every interval."""
        slow = (k % self.slow_every == 0)
        moves = []; self.act_ms_r = []
        if slow:
            lo = max(0, k - self.slow_every + 1)
            win = np.vstack([self.flags[lo:k].astype(bool), flagged[None]]) if k > lo else flagged[None]
            self.snap = win.any(axis=0)                       # flagged at any time in the window: avoid as a backup
            self.snap_pers = win.mean(axis=0) >= 0.5          # flagged in >= half the window: persistent
            self.msgs_longhaul += self.n_summary
        cap = max(1, MAX_MOVES // self.n_reg)
        for r in range(self.n_reg):
            t0_ = time.perf_counter()
            view = np.where(self.own[r], flagged, self.snap)                      # used to pick a clean backup
            trig = np.where(self.own[r], flagged, self.snap_pers) if self.policy == 'regional_v2' else view   # what may trigger a move
            self.last_flag_t[np.flatnonzero(view)] = now_s
            mine = [p for p in self.paths if self.region_of[p[0]] == r]
            cands = []
            for p in mine:
                if now_s - self.last_pair_change.get(p, -1e9) < HOLD_PAIR_S or self.pair_pk.get(p, 0) <= 0:
                    continue
                bad = [i for i in (self.dir_of[(x, y)] for x, y in zip(self.paths[p][:-1], self.paths[p][1:])) if trig[i]]
                if bad:
                    cands.append((max(float(np.nan_to_num(f['f_util'][i])) for i in bad), p, bad[0]))
            cands.sort(reverse=True)
            used = 0
            for _, p, i in cands:
                if used >= cap:
                    break
                best = next((c for c in self.cand[p] if c != self.paths[p] and
                             not any(view[self.dir_of[(x, y)]] for x, y in zip(c[:-1], c[1:]))), None)
                if best is None:
                    continue
                old = self.paths[p]; t = time.perf_counter(); self._install(p, best, r)
                moves.append(dict(kind='reroute', k=k, t_s=round(now_s, 3), trigger_dir=int(i), pair=list(p), old=old, new=best,
                                  region=r, install_ms=round((time.perf_counter() - t) * 1000, 2)))
                self.last_pair_change[p] = now_s; used += 1
            if slow:
                for p in mine:
                    if self.paths[p] == self.orig[p] or now_s - self.last_pair_change.get(p, -1e9) < REVERT_S:
                        continue
                    od = [self.dir_of[(x, y)] for x, y in zip(self.orig[p][:-1], self.orig[p][1:])]
                    if all(now_s - self.last_flag_t[j] >= REVERT_S for j in od):
                        old = self.paths[p]; t = time.perf_counter(); self._install(p, self.orig[p], r)
                        moves.append(dict(kind='revert', k=k, t_s=round(now_s, 3), trigger_dir=-1, pair=list(p), old=old,
                                          new=list(self.orig[p]), region=r, install_ms=round((time.perf_counter() - t) * 1000, 2)))
                        self.last_pair_change[p] = now_s
            self.act_ms_r.append((time.perf_counter() - t0_) * 1000)
        return moves

    def run(self):
        self.t0_ns = int(self.t0 * NS)
        for k in range(self.K):
            due = self.t0 + (k + 1) * INTERVAL_S + LAG_S
            d = due - time.time()
            if d > 0:
                time.sleep(d)
            try:
                t1 = time.perf_counter(); c0 = time.thread_time()
                self._ingest()
                f = self._interval(k)
                feats = self.eng.step(f)
                t2 = time.perf_counter(); meas_ms = (t2 - t1) * 1000
                rq2 = self.policy in RQ2_POLICIES
                if rq2:
                    if self.policy == 'central':
                        self.msgs_longhaul += self.n_nodes      # every switch reports to the one central agent
                    else:
                        self.msgs_local += self.n_nodes         # every switch reports to its regional agent
                    time.sleep(self.ctl_rtt_s)                  # emulated control-channel round trip (report up, rules down)
                    t2 = time.perf_counter()
                flagged = self._detect(k, f, feats)
                t3 = time.perf_counter()
                now_s = (k + 1) * INTERVAL_S
                if now_s < WARMUP_S:
                    moves = []
                elif self.policy in ('regional', 'regional_nofast', 'regional_v2'):
                    moves = self._act_regional(k, now_s, flagged, f)
                else:
                    moves = self._act(k, now_s, flagged, f)
                t4 = time.perf_counter()
                self.flags[k] = flagged
                for c in F_COLS:
                    self.fhist[c][k] = f[c]
                self.moves += moves
                st = dict(k=k, late_ms=round((time.time() - due) * 1000, 1), measure_ms=round(meas_ms, 2),
                          detect_ms=round((t3 - t2) * 1000, 2), act_ms=round((t4 - t3) * 1000, 2),
                          n_flagged=int(flagged.sum()), n_moves=len(moves), cpu_ms=round((time.thread_time() - c0) * 1000, 2))
                if rq2:      # decision latency = emulated control round trip + compute of the slowest agent (agents run in parallel)
                    if self.policy == 'central':
                        comp = (t3 - t2) * 1000 + (t4 - t3) * 1000
                    else:
                        comp = max((a + b for a, b in zip(self.det_ms_r, self.act_ms_r or [0.0] * self.n_reg)), default=0.0)
                    st.update(ctl_rtt_ms=round(self.ctl_rtt_ms, 2), lat_ms=round(self.ctl_rtt_ms + comp, 2))
                self.steps.append(st)
            except Exception as e:     # never let the controller crash the run
                import traceback
                self.errors.append(f'k={k}: {type(e).__name__}: {e} {traceback.format_exc()[-300:]}')

    def save(self, out):
        np.savez_compressed(out / 'controller_flags.npz', flags=self.flags, pred=self.pred, **{'live_' + c: v for c, v in self.fhist.items()})
        write_json(out / 'controller_log.json', {'policy': self.policy, 'moves': self.moves, 'steps': self.steps,
                                                 'n_flowmods': self.n_flowmods, 'errors': self.errors[:200],
                                                 'rq2': dict(n_flowmods_remote=self.fm_remote, msgs_longhaul=self.msgs_longhaul,
                                                             msgs_local=self.msgs_local, ctl_rtt_ms=self.ctl_rtt_ms, n_regions=self.n_reg,
                                                             maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                                                 'settings': dict(LAG_S=LAG_S, HOLD_DIR_S=HOLD_DIR_S, HOLD_PAIR_S=HOLD_PAIR_S,
                                                                  REVERT_S=REVERT_S, MAX_MOVES=MAX_MOVES,
                                                                  DELAY_RULE_MS=DELAY_RULE_MS, LOSS_RULE=LOSS_RULE,
                                                                  model=str(MODEL) if self.policy == 'model' else None)})


def pin_probes(net):
    """Probes between neighbours always use the direct link (so link delay stays measurable after reroutes)."""
    per = {i: [] for i in net.switches}
    for d in net.dirs:
        u, v = d['src'], d['dst']
        m = f'table=0,priority=200,udp,nw_src={host_ip(u)},nw_dst={host_ip(v)},tp_dst={PROBE_PORT}'
        per[u].append(f'{m},actions=output:{net.iface[(u, v)]["port"]}')
        per[v].append(f'{m},actions=output:{net.host_port[v]}')
    for i, rules in per.items():
        p = f'/tmp/ca_pin_{net.switches[i].name}.txt'
        Path(p).write_text('\n'.join(rules) + '\n')
        r = ofctl('add-flows', net.switches[i].name, p)
        if r.returncode != 0:
            raise RuntimeError(f'probe pin rules failed on {net.switches[i].name}: {r.stderr}')


def run(cfg, policy, build=True):
    from run_one import machine_cfg, versions
    assert policy in POLICIES + RQ2_POLICIES
    rid = f'{run_id(cfg)}__{policy}'
    out = LIVE_DIR / rid
    if (out / 'DONE').exists():
        print(f'{rid}: already done'); return out
    if out.exists():
        shutil.rmtree(out)
    raw = out / 'raw'; raw.mkdir(parents=True)
    mc = machine_cfg()
    if not SANDBOX and not mc.get('all_pass'):
        raise SystemExit('Refusing to run: smoke_test.py has not passed on this machine.')
    if policy == 'model_robust' and not (ROBUST_DIR / 'model.txt').exists():
        raise SystemExit(f'Robust model missing: {ROBUST_DIR}/model.txt')
    if (policy == 'model' or policy in RQ2_POLICIES) and not MODEL.exists():
        raise SystemExit(f'Model file missing: {MODEL}')
    topo = load_topology(cfg['topology'])
    net = Network(topo, normal_dist=mc.get('netem_normal_distribution', True))
    t_build = time.time(); procs, log = [], []
    ctl = None
    try:
        net.build()
        pin_probes(net)
        desc = net.describe()
        sched = make_schedule(cfg, topo, net.paths, net.dirs)
        write_json(out / 'schedule.json', sched)
        nb = {i: sorted({l['b'] for l in topo['links'] if l['a'] == i} | {l['a'] for l in topo['links'] if l['b'] == i})
              for i in net.hosts}
        t0 = time.time() + START_DELAY_S
        for i, h in net.hosts.items():
            hs = {'tcp_flows': [f for f in sched['tcp_flows'] if f['src'] == i],
                  'udp_flows': [f for f in sched['udp_flows'] if f['src'] == i]}
            sp = raw / f'sched_{i}.json'; sp.write_text(json.dumps(hs))
            pcfg = {'id': i, 'peers': [{'id': j, 'ip': host_ip(j)} for j in nb[i]], 't0': t0, 'duration': RUN_S,
                    'hz': PROBE_HZ, 'port': PROBE_PORT, 'out': str(raw)}
            acfg = {'id': i, 't0': t0, 'duration': RUN_S, 'out': str(raw), 'schedule': str(sp),
                    'tcp_port': TCP_PORT, 'udp_port': UDP_PORT}
            procs.append(h.popen(['python3', str(HERE / 'live_prober.py'), json.dumps(pcfg)],
                                 stdout=subprocess.DEVNULL, stderr=open(raw / f'prober_err_{i}.txt', 'w')))
            procs.append(h.popen(['python3', str(HERE.parent / 'datagen' / 'agent.py'), json.dumps(acfg)],
                                 stdout=subprocess.DEVNULL, stderr=open(raw / f'agent_err_{i}.txt', 'w')))
        intfs = [d['egress_intf'] for d in net.dirs] + [d['ingress_intf'] for d in net.dirs]
        col = Collector(intfs, [net.switches[i].name for i in sorted(net.switches)], t0 - 1.0, t0 + RUN_S + 1.0)
        col.start()
        ctl = Controller(net, topo, policy, raw, t0, col)
        ctl.start()
        actions = []
        for e in sched['noise']:
            actions.append((e['start'], 'noise_on', e)); actions.append((e['end'], 'noise_off', e))
        for e in sched['events']:
            if e['type'] == 'link_down':
                actions.append((e['start'], 'link_down', e)); actions.append((e['end'], 'link_up', e))
        actions.sort(key=lambda a: a[0])

        def scheduler():
            for at, kind, e in actions:
                if at > RUN_S + 0.5:
                    continue
                d = t0 + at - time.time()
                if d > 0:
                    time.sleep(d)
                a = time.time_ns()
                if kind == 'noise_on':
                    rc = net.set_noise(e['link_id'], e['jitter_ms'], e['loss_pct'], e.get('extra_delay_ms', 0.0))
                elif kind == 'noise_off':
                    rc = net.set_noise(e['link_id'], 0.0, 0.0)
                elif kind == 'link_down':
                    rc = net.set_link_state(e['link_id'], False)
                else:
                    rc = net.set_link_state(e['link_id'], True)
                b = time.time_ns()
                log.append({'kind': kind, 'link_id': e['link_id'], 'planned_s': at, 'start_ns': a, 'ts_ns': b, 'rc': rc,
                            'jitter_ms': e.get('jitter_ms', 0.0), 'loss_pct': e.get('loss_pct', 0.0),
                            'extra_delay_ms': e.get('extra_delay_ms', 0.0)})
        st = threading.Thread(target=scheduler, daemon=True); st.start()
        print(f'{rid}: running {RUN_S} s with controller "{policy}"', flush=True)
        time.sleep(max(0, t0 + RUN_S + 4.0 - time.time()))
        st.join(timeout=5); ctl.join(timeout=10); col.join()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        col.save(raw)
        ctl.save(out)
        write_json(out / 'actions_log.json', log)
        write_json(out / 'meta.json', {
            'run_id': rid, 'config': cfg, 'policy': policy, 'split': 'live', 't0_ns': int(t0 * 1e9), 'run_s': RUN_S,
            'sandbox': SANDBOX, 'userspace_ovs': USERSPACE_OVS, 'machine_config': mc, 'versions': versions(),
            'network': desc, 'build_and_run_wall_s': round(time.time() - t_build, 1),
            'topology_name': topo['name'], 'topology_source': topo['source']})
        (out / 'DONE').write_text('ok')
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        net.stop()
    for f in glob.glob(str(raw / 'live_*.csv')):
        os.remove(f)                       # same data is in probes_*.csv.gz
    if build:
        from build_intervals import build_run
        build_run(out, quiet=True)
    n = len([m for m in ctl.moves if m['kind'] == 'reroute'])
    print(f'{rid}: done, {n} reroutes, {ctl.n_flowmods} flow-mods, controller errors {len(ctl.errors)}', flush=True)
    return out


if __name__ == '__main__':
    a = sys.argv[1:]
    cfg = {'topology': a[0], 'load': int(a[1]), 'noise': a[2], 'transport': a[3], 'event': a[4], 'seed': int(a[5])}
    run(cfg, a[6])
