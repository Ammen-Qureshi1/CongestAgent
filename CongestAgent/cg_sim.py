"""cg_sim.py - flow-level simulator for CongestAgent (Steps 5b/5c/6 support).

What it simulates (every 250 ms interval, per directed link, 10 Mbps, 150 kB queue):
  * traffic: UDP ON/OFF + TCP (CUBIC / BBR) aggregates, optional bursts and a link failure
  * a fluid queue per link  -> real congestion (queue delay, drops)
  * non-congestive noise: random loss, jitter, delay shift (episodes on 30% of links)
  * the SAME labels as the dataset (congestion / degradation / normal / uncertain)
  * a perception emulator that reproduces the real classifier's confusion matrix
Everything random about the WORLD (traffic, noise, failures) is drawn once from the scenario seed, so
every method sees exactly the same world (paired comparisons).
"""
import json, os, math
from dataclasses import dataclass
from itertools import islice
from collections import deque
import numpy as np
import networkx as nx

DT = 0.25                      # seconds per interval
CAP = 10e6                     # link capacity, bit/s
QBITS = 150000 * 8             # queue limit, bits
QDELAY_MAX = QBITS / CAP       # 0.12 s
KP = 3                         # candidate paths per aggregate
LAB_NORMAL, LAB_DEG, LAB_CONG, LAB_UNC = 0, 1, 2, 3

NSFNET_EDGES = [(1, 2), (1, 3), (1, 8), (2, 3), (2, 4), (3, 6), (4, 5), (4, 11), (5, 6), (5, 7), (6, 10),
                (6, 14), (7, 8), (8, 9), (9, 10), (9, 12), (9, 13), (11, 12), (11, 13), (12, 14), (13, 14)]

CALIB_DEFAULT = dict(tcp_sigma=0.8, tcp_rho=0.9, udp_mean_on=4.0, m_cubic=1.2, m_bbr=0.8,
                     k_cubic_loss=10.0, k_bbr_loss=1.5, transient_gain=0.5, transient_intervals=4,
                     cubic_jitter_k=0.003, bbr_jitter_k=0.0008)


def load_calib(path=None):
    c = dict(CALIB_DEFAULT)
    if path and os.path.exists(path):
        c.update({k: v for k, v in json.load(open(path)).items() if k in c})
    return c


# ----------------------------------------------------------------------------- topology
def _parse_topology_json(obj):
    """Accepts several common json layouts. Returns (nodes, edges, prop_ms or None)."""
    if "topology" in obj and isinstance(obj["topology"], dict):
        obj = obj["topology"]
    nodes_raw = obj.get("nodes") or obj.get("node") or obj.get("vertices")
    links_raw = obj.get("links") or obj.get("edges") or obj.get("link")
    if isinstance(nodes_raw, dict):
        nodes_raw = list(nodes_raw.keys())
    if isinstance(links_raw, dict):
        links_raw = list(links_raw.values())

    def nid(x):
        if isinstance(x, dict):
            for k in ("id", "name", "label", "node"):
                if k in x:
                    return x[k]
        return x

    nodes = [nid(n) for n in nodes_raw] if nodes_raw else None
    pairs = [("source", "target"), ("src", "dst"), ("from", "to"), ("a", "b"), ("u", "v"), ("node1", "node2"),
             ("src_node", "dst_node")]
    edges, delays, seen = [], [], set()
    for l in links_raw:
        a = b = None
        if isinstance(l, (list, tuple)):
            a, b = l[0], l[1]
        elif isinstance(l, dict):
            for p, q in pairs:
                if p in l and q in l:
                    a, b = l[p], l[q]
                    break
            if a is None and "nodes" in l:
                a, b = l["nodes"][:2]
        if a is None:
            raise ValueError(f"cannot read link entry: {l}")
        a, b = nid(a), nid(b)
        key = (min(str(a), str(b)), max(str(a), str(b)))
        if a == b or key in seen:
            continue
        seen.add(key)
        edges.append((a, b))
        d = None
        if isinstance(l, dict):
            for k in ("prop_delay_ms", "delay_ms", "propagation_delay_ms", "delay"):
                if k in l:
                    d = float(l[k])
                    break
            if d is None:
                for k in ("length_km", "distance_km", "km"):
                    if k in l:
                        d = float(np.clip(float(l[k]) / 200.0, 2, 10))
                        break
        delays.append(d)
    if nodes is None:
        nodes = sorted({x for e in edges for x in e}, key=str)
    prop = None if any(d is None for d in delays) else delays
    return nodes, edges, prop


def _synthetic_graph(n, m, seed):
    rng = np.random.default_rng(seed)
    edges = {(i, int(rng.integers(0, i))) for i in range(1, n)}
    while len(edges) < m:
        a, b = rng.integers(0, n, 2)
        if a != b:
            edges.add((int(min(a, b)), int(max(a, b))))
    return list(range(n)), [tuple(e) for e in edges]


class Topology:
    def __init__(self, name, nodes, edges, prop_ms=None, seed=0, n_regions=4):
        self.name = name
        self.nodes = list(nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        E = [(self.idx[a], self.idx[b]) for a, b in edges]
        self.n_nodes, self.n_edges, self.n_links = len(self.nodes), len(E), 2 * len(E)
        rng = np.random.default_rng(seed + 12345)
        prop_ms = rng.uniform(2, 10, len(E)) if prop_ms is None else np.clip(np.asarray(prop_ms, float), 2, 10)
        self.edges = E
        L = self.n_links
        self.link_src, self.link_dst = np.zeros(L, int), np.zeros(L, int)
        self.prop = np.zeros(L)
        self.link_id = {}
        for k, (u, v) in enumerate(E):
            self.link_src[2 * k], self.link_dst[2 * k] = u, v
            self.link_src[2 * k + 1], self.link_dst[2 * k + 1] = v, u
            self.link_id[(u, v)], self.link_id[(v, u)] = 2 * k, 2 * k + 1
            self.prop[2 * k] = self.prop[2 * k + 1] = prop_ms[k] / 1000.0
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_nodes))
        self.G.add_edges_from(E)
        assert nx.is_connected(self.G), "topology must be connected"
        self.n_regions = min(n_regions, self.n_nodes)
        self.region = self._partition(self.n_regions)
        self.link_region = self.region[self.link_src]
        self.region_nbrs = {r: set() for r in range(self.n_regions)}
        for u, v in E:
            if self.region[u] != self.region[v]:
                self.region_nbrs[self.region[u]].add(self.region[v])
                self.region_nbrs[self.region[v]].add(self.region[u])
        self._paths = {}

    def _partition(self, R):
        """Regions = nearest of R seed nodes. Tries many seed sets and keeps the most balanced split."""
        G = self.G
        d = dict(nx.all_pairs_shortest_path_length(G))
        rng = np.random.default_rng(99)
        best, best_score = None, 1e9
        for _ in range(300):
            seeds = [int(x) for x in rng.choice(G.number_of_nodes(), R, replace=False)]
            reg = np.array([int(np.argmin([d[s][n] + 1e-3 * i for i, s in enumerate(seeds)])) for n in G.nodes])
            sizes = np.bincount(reg, minlength=R)
            score = sizes.max() - sizes.min()
            conn = all(nx.is_connected(G.subgraph(np.where(reg == r)[0].tolist())) for r in range(R)
                       if (reg == r).any())
            if conn and score < best_score:
                best, best_score = reg, score
        return best

    def k_paths(self, s, t):
        if (s, t) not in self._paths:
            ps = list(islice(nx.shortest_simple_paths(self.G, s, t), KP))
            while len(ps) < KP:
                ps.append(ps[-1])
            self._paths[(s, t)] = ps
        return self._paths[(s, t)]


def get_topology(name, topo_dir=None, seed=0):
    """name: 'nsfnet' or 'geant'. Reads <topo_dir>/<name>.json if it exists, else uses a built-in layout."""
    if topo_dir:
        p = os.path.join(topo_dir, f"{name}.json")
        if os.path.exists(p):
            try:
                nodes, edges, prop = _parse_topology_json(json.load(open(p)))
                return Topology(name, nodes, edges, prop, seed)
            except Exception as e:
                print(f"[warn] could not read {p} ({e}); using built-in {name}")
    if name == "nsfnet":
        return Topology("nsfnet(builtin)", list(range(1, 15)), NSFNET_EDGES, None, seed)
    nodes, edges = _synthetic_graph(22, 36, 7)
    return Topology("geant(SYNTHETIC stand-in, 22 nodes 36 links)", nodes, edges, None, seed)


# ----------------------------------------------------------------------------- scenario
@dataclass
class Scenario:
    topo: str = "nsfnet"
    load: float = 0.6
    noise: str = "none"          # none | loss0.5 | loss1 | loss2 | jitter5/10/20/40 | shift40 | shift80
    transport: str = "mixed"     # mixed | cubic
    event: str = "none"          # none | burst | failure
    seed: int = 0
    duration: float = 120.0
    n_pairs: int = 0             # 0 -> 30 (nsfnet) / 45 (geant)

    def key(self):
        return f"{self.topo}_L{int(self.load * 100)}_{self.noise}_{self.transport}_{self.event}"


def parse_noise(s):
    if s == "none":
        return "none", 0.0
    for kind in ("loss", "jitter", "shift"):
        if s.startswith(kind):
            return kind, float(s[len(kind):])
    raise ValueError(s)


# ----------------------------------------------------------------------------- perception emulator
class PerceptionEmu:
    """Turns TRUE link labels into the labels the classifier would output.
    mode: 'labels' (3 classes, confusion matrix cm), 'merged' (degradation counted as congestion: a classifier
    that only sees symptoms), 'oracle', 'none' (no perception: B2).
    Errors are persistent: if the true label did not change, the previous output is kept with prob. rho."""

    def __init__(self, cm, mode, rng, rho=0.7):
        self.mode, self.rng, self.rho = mode, rng, rho
        self.prev_true = self.prev_obs = None
        if cm is not None:
            P = np.asarray(cm, float) + 1e-9
            self.cdf = np.cumsum(P / P.sum(1, keepdims=True), 1)

    def observe(self, lab, qfrac, noise_on):
        L = len(lab)
        if self.mode == "none":
            z = np.zeros(L, bool)
            return z, z
        eff = lab.copy()
        unc = lab == LAB_UNC
        if unc.any():
            eff[unc] = np.where(qfrac[unc] >= 0.35, 2, np.where(noise_on[unc], 1, 0))
        if self.mode == "oracle":
            obs = eff
        else:
            u = self.rng.random(L)
            samp = np.minimum((u[:, None] > self.cdf[eff]).sum(1), 2)
            if self.prev_true is None:
                obs = samp
            else:
                keep = (self.prev_true == eff) & (self.rng.random(L) < self.rho)
                obs = np.where(keep, self.prev_obs, samp)
        self.prev_true, self.prev_obs = eff, obs
        if self.mode == "merged":
            return obs >= 1, np.zeros(L, bool)
        return obs == 2, obs == 1


def cm_from_metrics(path, splits):
    """Sum the confusion matrices (rows=true, cols=pred) of the given splits in a metrics.json."""
    m = json.load(open(path))
    cm = np.zeros((3, 3))
    for s in splits:
        cm += np.array(m[s]["confusion_matrix(rows=true, cols=pred)"], float)
    return cm


# ----------------------------------------------------------------------------- the network
def weighted_percentile(x, w, q):
    if len(x) == 0:
        return float("nan")
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(x))
    return float(x[np.searchsorted(c, q * c[-1])])


class Net:
    def __init__(self, topo, scen, calib=None, perception=None, obs_noise=0.03, summary_delay=1,
                 fast_loop=True, ecmp=False, obs_seed_salt=0, warmup_s=20.0, scope="regional"):
        self.topo, self.scen = topo, scen
        self.c = calib or load_calib()
        self.percep = perception
        self.obs_noise, self.summary_delay, self.fast_loop = obs_noise, summary_delay, fast_loop
        self.scope = scope
        self.rng_obs = np.random.default_rng(scen.seed * 1009 + 17 + obs_seed_salt)
        rng = np.random.default_rng(scen.seed)        # world rng: identical for every method
        L, T = topo.n_links, int(round(scen.duration / DT))
        self.L, self.T, self.warm_n = L, T, int(round(warmup_s / DT))
        self.prop = topo.prop

        # --- aggregates (source-destination pairs) and candidate paths
        K = scen.n_pairs or (45 if scen.topo == "geant" else 30)
        pairs = set()
        while len(pairs) < K:
            s, d = rng.integers(0, topo.n_nodes, 2)
            if s != d:
                pairs.add((int(s), int(d)))
        self.pairs = sorted(pairs)
        self.K = K
        w = rng.lognormal(0, 0.8, K)
        self.P = np.zeros((K, KP, L))
        self.hops = np.zeros((K, KP))
        self.base_delay = np.zeros((K, KP))
        for i, (s, d) in enumerate(self.pairs):
            for p, path in enumerate(topo.k_paths(s, d)):
                ids = [topo.link_id[(a, b)] for a, b in zip(path[:-1], path[1:])]
                self.P[i, p, ids] = 1.0
                self.hops[i, p] = len(ids)
                self.base_delay[i, p] = self.prop[ids].sum()
        self.P2 = self.P.reshape(K * KP, L)
        self.agg_region = topo.region[[s for s, _ in self.pairs]]
        load0 = (w[:, None] * self.P[:, 0, :]).sum(0)
        b = w * (scen.load * CAP / load0.max())       # mean rate per aggregate (bit/s)

        # --- traffic traces (drawn once)
        c = self.c
        p_flip = 1 - math.exp(-DT / c["udp_mean_on"])
        on = np.zeros((T, K), bool)
        st = rng.random(K) < 0.5
        for t in range(T):
            st = st ^ (rng.random(K) < p_flip)
            on[t] = st
        udp = on * b[None, :]                           # ON rate = b, duty 0.5 -> mean 0.5 b
        z = rng.normal(size=K)
        rho = c["tcp_rho"]
        tcp = np.zeros((T, K))
        sg = c["tcp_sigma"]
        for t in range(T):
            z = rho * z + math.sqrt(1 - rho ** 2) * rng.normal(size=K)
            tcp[t] = 0.5 * b * np.exp(sg * z - sg ** 2 / 2)
        share_c = 1.0 if scen.transport == "cubic" else 0.5
        self.dem_cub, self.dem_bbr = tcp * share_c, tcp * (1 - share_c)
        if scen.event == "burst":
            for _ in range(max(1, int(scen.duration / 6))):
                t0 = int(rng.integers(self.warm_n // 2, T - 8))
                n = int(math.ceil(rng.uniform(0.3, 1.5) / DT))
                udp[t0:t0 + n, int(rng.integers(0, K))] += rng.uniform(1.5, 3.0) * CAP
        self.dem_udp = udp

        # --- noise episodes (both directions of a random 30% of links)
        kind, lvl = parse_noise(scen.noise)
        self.noise_kind = kind
        n_on = np.zeros((T, topo.n_edges), bool)
        if kind != "none":
            chosen = rng.choice(topo.n_edges, max(1, int(round(0.3 * topo.n_edges))), replace=False)
            for e in chosen:
                t, state = 0, bool(rng.random() < 0.5)
                while t < T:
                    n = int(rng.uniform(10, 30) / DT)
                    n_on[t:t + n, e] = state
                    t, state = t + n, not state
        on_l = np.repeat(n_on, 2, axis=1)               # T, L
        self.loss_noise = on_l * (lvl / 100.0 if kind == "loss" else 0.0)
        self.jit = on_l * (lvl / 1000.0 if kind == "jitter" else 0.0)
        self.shift = on_l * (lvl / 1000.0 if kind == "shift" else 0.0)
        self.noise_on_tr = on_l

        # --- link failure
        self.down = np.zeros((T, L), bool)
        if scen.event == "failure":
            used = np.where(self.P[:, 0, :].sum(0) > 0)[0] // 2
            e = int(rng.choice(np.unique(used)))
            t0 = int(rng.uniform(30, max(31, scen.duration - 30)) / DT)
            self.down[t0:t0 + int(rng.uniform(10, 20) / DT), 2 * e:2 * e + 2] = True

        # --- routing state
        self.prim = np.zeros(K, int)
        self.back = np.ones(K, int)
        self.eff = np.zeros(K, int)
        self.ecmp = ecmp
        self.W = np.zeros((K, KP))
        if ecmp:                                        # B1: equal split over equal-hop shortest paths
            for i in range(K):
                m = (self.hops[i] == self.hops[i, 0])
                self.W[i, m] = 1.0 / m.sum()
        else:
            self.W[np.arange(K), self.eff] = 1.0

        # --- dynamic state
        self.t = 0
        self.q = np.zeros(L)
        self.qhist = [np.zeros(L), np.zeros(L)]
        self.trans_left = np.zeros(K, int)
        self.rate_ema = b.copy()
        self.lab = np.zeros(L, int)
        self.ld = self.prop.copy()
        self.loss_true = np.zeros(L)
        z0 = np.zeros(L)
        self.hist = deque([dict(util=z0, delay=self.prop.copy(), loss=z0, cong=z0.astype(bool),
                                deg=z0.astype(bool))] * 6, maxlen=6)
        # --- counters / metrics
        self.msg_flowmod = 0
        self.fm_region = np.zeros(topo.n_regions)
        self.rc_total = self.rc_false = self.rc_unjust = 0
        self.rc_fast = self.rc_slow = 0
        self.last_moved_deg = np.zeros(K, bool)
        self.acc = dict(thr=0.0, off=0.0, lost=0.0, jain=0.0, jain_w=0.0, n=0)
        self.dl_x, self.dl_w = [], []
        self.cyc_thr = self.cyc_off = self.cyc_lost = 0.0
        self.cyc_x, self.cyc_w = [], []
        self.cm_obs = np.zeros((3, 3))                  # true (rows) vs observed (cols) label counts
        self.cong_int = self.deg_int = self.norm_int = self.unc_int = 0

    # ------------------------------------------------------------------ routing
    def _counting(self):
        return self.t >= self.warm_n

    def _reroute(self, ids, new_eff, kind):
        """Switch aggregates `ids` to path index new_eff (array). Counts and classifies each change."""
        moved = np.zeros(self.K, bool)
        for i, ne in zip(ids, new_eff):
            old = self.eff[i]
            if ne == old:
                continue
            links = self.P[i, old] > 0
            any_cong = bool((self.lab[links] == LAB_CONG).any())
            any_deg = bool((self.lab[links] == LAB_DEG).any())
            moved[i] = (not any_cong) and any_deg
            self.eff[i] = ne
            self.W[i] = 0.0
            self.W[i, ne] = 1.0
            self.trans_left[i] = int(self.c["transient_intervals"])
            if self._counting():
                self.rc_total += 1
                self.rc_false += int((not any_cong) and any_deg)
                self.rc_unjust += int(not any_cong)
                if kind == "fast":
                    self.rc_fast += 1
                else:
                    self.rc_slow += 1
            self.msg_flowmod += 1 if kind == "fast" else int(self.hops[i, ne])
            self.fm_region[self.agg_region[i]] += 1 if kind == "fast" else int(self.hops[i, ne])
        return moved

    def apply_action(self, i, prim, back):
        """Slow loop: install primary + backup for aggregate i. Returns (changed, moved_from_degraded)."""
        pair_changed = (self.prim[i] != prim) or (self.back[i] != back)
        self.prim[i], self.back[i] = prim, back
        if pair_changed:
            self.msg_flowmod += int(self.hops[i, back])
            self.fm_region[self.agg_region[i]] += int(self.hops[i, back])
        old = self.eff[i]
        moved = self._reroute([i], [prim], "slow")
        return bool(self.eff[i] != old), bool(moved[i])

    def apply_actions(self, prims, backs):
        for i in range(self.K):
            self.apply_action(i, int(prims[i]), int(backs[i]))

    def _fast_loop(self):
        cong = self.hist[-1]["cong"]
        view = self._scoped(np.arange(self.K), "cong")
        pr = self.P[np.arange(self.K), self.prim]
        bk = self.P[np.arange(self.K), self.back]
        prim_bad = ((pr * view).sum(1) > 0) & (self.eff == self.prim)
        back_ok = ~((bk * view).sum(1) > 0)
        sw = np.where(prim_bad & back_ok & (self.back != self.prim))[0]
        if len(sw):
            self._reroute(sw, self.back[sw], "fast")

    # ------------------------------------------------------------------ one interval of physics
    def step(self):
        t, K, L = self.t, self.K, self.L
        udp, cub, bbr = self.dem_udp[t], self.dem_cub[t], self.dem_bbr[t]
        tot = udp + cub + bbr
        load = (self.W * tot[:, None]).ravel() @ self.P2
        down = self.down[t]
        qprev = self.q
        qraw = qprev + (load - CAP) * DT
        q = np.clip(qraw, 0, QBITS)
        dropf = np.minimum(np.maximum(qraw - QBITS, 0) / np.maximum(load * DT, 1e-9), 1.0)
        dropf = np.where(down, 1.0, dropf)
        q = np.where(down, 0.0, q)
        qavg = 0.5 * (qprev + q)
        qfrac = qavg / QBITS
        self.q = q
        ln, sig, sh = self.loss_noise[t], self.jit[t], self.shift[t]
        noise_on = self.noise_on_tr[t] & (self.noise_kind != "none")
        ld = self.prop + qavg / CAP + sh
        self.ld, self.loss_true = ld, np.minimum(dropf + ln, 1.0)

        # --- retention of each traffic class along each candidate path
        c = self.c
        ret_u = (1 - dropf) * (1 - ln)
        ret_c = np.clip(1 - c["m_cubic"] * dropf, 0, 1) * (1 - ln)
        ret_b = np.clip(1 - c["m_bbr"] * dropf, 0, 1) * (1 - ln)

        def pathret(r):
            return np.exp(self.P2 @ np.log(np.maximum(r, 1e-6))).reshape(K, KP)

        retU = (self.W * pathret(ret_u)).sum(1)
        retC = (self.W * pathret(ret_c)).sum(1)
        retB = (self.W * pathret(ret_b)).sum(1)
        pe = (self.W * (1 - pathret(1 - ln))).sum(1)                     # end-to-end random loss
        jsum = (self.W * (self.P2 @ sig).reshape(K, KP)).sum(1) * 1000.0  # ms of jitter on the path
        gc = np.clip(1 - c["k_cubic_loss"] * pe, 0, 1) * np.clip(1 - c["cubic_jitter_k"] * jsum, 0.3, 1)
        gb = np.clip(1 - c["k_bbr_loss"] * pe, 0, 1) * np.clip(1 - c["bbr_jitter_k"] * jsum, 0.3, 1)
        trans = np.where(self.trans_left > 0, c["transient_gain"], 1.0)
        self.trans_left = np.maximum(self.trans_left - 1, 0)
        xc, xb = retC * gc * trans, retB * gb * trans
        delivered = udp * retU + cub * xc + bbr * xb
        pd = (self.W * (self.P2 @ ld).reshape(K, KP)).sum(1)

        # --- true labels (same rule as the dataset)
        p1, p2 = self.qhist
        cong = ((qfrac > 0.5) & (p1 > 0.5) & (p2 > 0.5)) | down
        deg = noise_on & (qfrac < 0.2) & ~cong
        norm = (~noise_on) & (qfrac < 0.2) & ~cong
        lab = np.full(L, LAB_UNC)
        lab[norm], lab[deg], lab[cong] = LAB_NORMAL, LAB_DEG, LAB_CONG
        self.qhist = [qfrac, p1]
        self.lab = lab

        # --- measurements the controller can see
        rng = self.rng_obs
        util_m = np.clip(np.where(down, 0.0, np.minimum(load, CAP) / CAP) + rng.normal(0, self.obs_noise, L), 0, 1)
        delay_m = np.maximum(ld + rng.normal(0, 1, L) * (0.55 * sig / math.sqrt(5) + 0.0003), self.prop)
        loss_m = np.clip(self.loss_true + rng.normal(0, 0.002, L), 0, 1)
        if self.percep is not None:
            cong_o, deg_o = self.percep.observe(lab, qfrac, noise_on)
        else:
            cong_o = deg_o = np.zeros(L, bool)
        self.hist.append(dict(util=util_m, delay=delay_m, loss=loss_m, cong=cong_o, deg=deg_o))
        self.rate_ema = 0.9 * self.rate_ema + 0.1 * tot

        self.cyc_thr += delivered.sum() * DT
        self.cyc_off += tot.sum() * DT
        self.cyc_lost += (tot * (1 - retU)).sum() * DT
        mm = delivered > 0
        self.cyc_x.append(pd[mm])
        self.cyc_w.append(delivered[mm])

        # --- metrics (after warm-up only)
        if self._counting():
            a = self.acc
            a["thr"] += delivered.sum() * DT
            a["off"] += tot.sum() * DT
            a["lost"] += (tot * (1 - retU)).sum() * DT
            both = (cub > 0) & (bbr > 0)
            if both.any():
                ww = (cub + bbr)[both]
                J = (xc + xb)[both] ** 2 / (2 * (xc ** 2 + xb ** 2)[both] + 1e-12)
                a["jain"] += (J * ww).sum()
                a["jain_w"] += ww.sum()
            a["n"] += 1
            m = delivered > 0
            self.dl_x.append(pd[m])
            self.dl_w.append(delivered[m])
            for tl, ob in ((0, 0), (1, 1), (2, 2)):
                sel = lab == tl
                if sel.any() and self.percep is not None and self.percep.prev_obs is not None:
                    self.cm_obs[tl] += np.bincount(self.percep.prev_obs[sel], minlength=3)[:3]
            self.norm_int += int(norm.sum())
            self.deg_int += int(deg.sum())
            self.cong_int += int(cong.sum())
            self.unc_int += int((lab == LAB_UNC).sum())
        self.t += 1
        if self.fast_loop:
            self._fast_loop()

    # ------------------------------------------------------------------ observations for the agents
    def _scoped(self, idx, key):
        """Per-aggregate view of link array `key`: own-region links live, other regions' links delayed."""
        cur = self.hist[-1][key].astype(float)
        old = self.hist[max(-len(self.hist), -1 - self.summary_delay)][key].astype(float)
        if getattr(self, "scope", "regional") == "central":
            return np.broadcast_to(cur, (len(idx), self.L))
        own = self.topo.link_region[None, :] == self.agg_region[np.asarray(idx)][:, None]
        return np.where(own, cur[None, :], old[None, :])

    def obs_for(self, idx, labels=True):
        idx = np.asarray(idx)
        n = len(idx)
        Pi, hops = self.P[idx], self.hops[idx]
        V = {k: self._scoped(idx, k) for k in ("util", "delay", "loss", "cong", "deg")}

        def ps(a):
            return np.einsum("npl,nl->np", Pi, a)

        exc = np.clip(ps(np.maximum(V["delay"] - self.prop[None, :], 0)) / QDELAY_MAX, 0, 3) / 3
        mu = (Pi * V["util"][:, None, :]).max(2)
        fc = ps(V["cong"]) / np.maximum(hops, 1) if labels else np.zeros((n, KP))
        fd = ps(V["deg"]) / np.maximum(hops, 1) if labels else np.zeros((n, KP))
        loss = np.minimum(1 - np.exp(ps(np.log(np.maximum(1 - V["loss"], 1e-6)))), 0.1) * 10
        isp = np.zeros((n, KP))
        isp[np.arange(n), self.prim[idx]] = 1
        f = np.stack([exc, mu, fc, fd, loss, hops / 10, self.base_delay[idx] / 0.05, isp], axis=2).reshape(n, 24)
        g = np.stack([V["cong"].mean(1) if labels else np.zeros(n), V["deg"].mean(1) if labels else np.zeros(n),
                      np.clip(self.rate_ema[idx] / CAP, 0, 2), (self.eff[idx] != self.prim[idx]).astype(float)], 1)
        return np.concatenate([f, g], 1).astype(np.float32)

    def true_cost(self, i, p):
        """Immediate cost of using candidate path p for aggregate i from the TRUE state (for the training reward)."""
        pl = self.P[i, p] > 0
        d = self.ld[pl].sum()
        ls = 1 - np.prod(1 - self.loss_true[pl])
        return d / 0.1 + 5.0 * ls

    def pop_cycle(self):
        """Realised outcome since the last call (used for the end-of-cycle reward)."""
        x = np.concatenate(self.cyc_x) if self.cyc_x else np.array([])
        w = np.concatenate(self.cyc_w) if self.cyc_w else np.array([])
        r = dict(thr=self.cyc_thr / max(self.cyc_off, 1e-9), loss=self.cyc_lost / max(self.cyc_off, 1e-9),
                 p95=weighted_percentile(x, w, 0.95) if len(x) else 0.0)
        self.cyc_thr = self.cyc_off = self.cyc_lost = 0.0
        self.cyc_x, self.cyc_w = [], []
        return r

    # ------------------------------------------------------------------ results
    def metrics(self):
        a = self.acc
        mins = max(a["n"] * DT / 60.0, 1e-9)
        x = np.concatenate(self.dl_x) if self.dl_x else np.array([])
        w = np.concatenate(self.dl_w) if self.dl_w else np.array([])
        return dict(
            throughput_mbps=a["thr"] / max(a["n"] * DT, 1e-9) / 1e6,
            delay_mean_ms=1000 * float((x * w).sum() / max(w.sum(), 1e-9)) if len(x) else float("nan"),
            delay_p95_ms=1000 * weighted_percentile(x, w, 0.95),
            loss_pct=100 * a["lost"] / max(a["off"], 1e-9),
            jain=a["jain"] / a["jain_w"] if a["jain_w"] > 0 else float("nan"),
            route_changes_per_min=self.rc_total / mins,
            false_reroute_rate=self.rc_false / self.rc_total if self.rc_total else 0.0,
            unjustified_reroute_rate=self.rc_unjust / self.rc_total if self.rc_total else 0.0,
            n_reroutes=self.rc_total, n_fast=self.rc_fast, n_slow=self.rc_slow,
            share_congestion=self.cong_int / max(self.L * a["n"], 1),
            share_degradation=self.deg_int / max(self.L * a["n"], 1),
        )
