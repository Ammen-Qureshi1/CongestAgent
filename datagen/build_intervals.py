"""
Turn one run's raw readings into the labelled dataset (one row per link direction
per 250 ms interval) plus flow-level files and an automatic quality report.

Column groups (see DATASHEET.md):
  meta_*    identifiers and run settings (never model inputs)
  static_*  fixed link attributes a controller knows (capacity, propagation delay)
  f_*       controller-observable measurements (model inputs)
  gt_*      ground truth from the emulator: queue backlog, drop causes, noise state,
            and the label (never model inputs)

Label rule, using the tbf backlog averaged over the 50 ms samples
in the interval, as a share of the queue limit:
  congestion  : backlog > 50% in this and the 2 previous intervals (priority)
  degradation : noise active for the whole interval and backlog < 20%
  normal      : no noise during the interval and backlog < 20%
  uncertain   : anything else (kept, but not for training); the reason is recorded
"""
import sys
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path

from common import (INTERVAL_S, WARMUP_S, PROBE_HZ, QUEUE_LIMIT_BYTES, CAPACITY_MBPS, PATHLOSS_WINDOW_S,
                    LABEL_HIGH, LABEL_LOW, LABEL_CONSEC, load_topology, read_json, write_json)

NS = 1_000_000_000
MIN_PKTS_PATHLOSS = 20


def _interp_cum(ts, vals, at):
    """Linear interpolation of a cumulative counter; NaN outside the sampled range."""
    if len(ts) < 2:
        return np.full(len(at), np.nan)
    return np.interp(at, ts, vals, left=np.nan, right=np.nan)


def _diff_at(ts, vals, edges):
    c = _interp_cum(ts, vals, edges)
    return np.diff(c)


def build_run(run_dir, quiet=False):
    run_dir = Path(run_dir)
    raw = run_dir / 'raw'
    meta = read_json(run_dir / 'meta.json')
    sched = read_json(run_dir / 'schedule.json')
    actions = read_json(run_dir / 'actions_log.json')
    cfg = meta['config']
    topo = load_topology(cfg['topology'])
    links = {l['id']: l for l in topo['links']}
    region = {n['id']: n['region'] for n in topo['nodes']}
    dirs = meta['network']['directions']
    t0, run_s = meta['t0_ns'], meta['run_s']
    K = int(round(run_s / INTERVAL_S))
    edges = t0 + (np.arange(K + 1) * INTERVAL_S * NS).astype(np.int64)
    mc = meta.get('machine_config', {})
    netem_counts_child = bool(mc.get('netem_counts_child_drops', True))

    ports = pd.read_csv(raw / 'ports.csv.gz')
    queues = pd.read_csv(raw / 'queues.csv.gz')
    flows = pd.read_csv(raw / 'flowstats.csv.gz')
    pgrp = {k: g.sort_values('ts_ns') for k, g in ports.groupby('intf')}
    qgrp = {k: g.sort_values('ts_ns') for k, g in queues.groupby(['intf', 'kind'])}

    # ---------------- probes ----------------
    tx = pd.concat([pd.read_csv(f).assign(src=int(Path(f).name.split('_')[-1].split('.')[0]))
                    for f in glob.glob(str(raw / 'probes_tx_*.csv.gz'))], ignore_index=True)
    rx = pd.concat([pd.read_csv(f).assign(dst=int(Path(f).name.split('_')[-1].split('.')[0]))
                    for f in glob.glob(str(raw / 'probes_rx_*.csv.gz'))], ignore_index=True)
    send_err = tx[tx.send_ns < 0]
    tx = tx[tx.send_ns > 0]
    pr = tx.merge(rx[['src', 'dst', 'seq', 'recv_ns']], on=['src', 'dst', 'seq'], how='left')
    pr['delay_ms'] = (pr.recv_ns - pr.send_ns) / 1e6
    pr['k'] = ((pr.send_ns - t0) // int(INTERVAL_S * NS)).astype(int)
    pr = pr[(pr.k >= 0) & (pr.k < K)]
    # reordering: received after a probe with a higher sequence number (same direction)
    rcv = pr.dropna(subset=['recv_ns']).sort_values(['src', 'dst', 'recv_ns'])
    rcv['reordered'] = rcv.seq < rcv.groupby(['src', 'dst']).seq.cummax()
    pr = pr.merge(rcv[['src', 'dst', 'seq', 'reordered']], on=['src', 'dst', 'seq'], how='left')
    pgb = pr.groupby(['src', 'dst', 'k'])
    pstats = pd.DataFrame({
        'sent': pgb.size(), 'recv': pgb.recv_ns.count(),
        'd_mean': pgb.delay_ms.mean(), 'd_min': pgb.delay_ms.min(), 'd_max': pgb.delay_ms.max(),
        'd_std': pgb.delay_ms.std(ddof=0), 'd_p50': pgb.delay_ms.median(),
        'reord': pgb.reordered.sum()})

    # ---------------- path loss from OpenFlow flow counters ----------------
    n_nodes = len(topo['nodes'])
    sw_name = {int(k): v for k, v in meta['network']['switch_names'].items()}
    fgrp = {k: g.sort_values('ts_ns') for k, g in flows.groupby(['switch', 'cookie'])}
    paths = {tuple(map(int, k.split('-'))): v for k, v in meta['network']['paths'].items()}
    end_edges = edges[1:]
    win_start = end_edges - int(PATHLOSS_WINDOW_S * NS)
    pair_loss, pair_pk = {}, {}
    for (a, b), path in paths.items():
        c = a * n_nodes + b + 1
        f1, f2 = fgrp.get((sw_name[a], c)), fgrp.get((sw_name[b], c))
        if f1 is None or f2 is None:
            continue
        first = _interp_cum(f1.ts_ns.values, f1.n_packets.values, end_edges) - \
            _interp_cum(f1.ts_ns.values, f1.n_packets.values, win_start)
        last = _interp_cum(f2.ts_ns.values, f2.n_packets.values, end_edges) - \
            _interp_cum(f2.ts_ns.values, f2.n_packets.values, win_start)
        with np.errstate(invalid='ignore', divide='ignore'):
            loss = np.clip((first - last) / first, 0.0, 1.0)
        loss[~(first >= MIN_PKTS_PATHLOSS)] = np.nan
        pair_loss[(a, b)] = loss
        pair_pk[(a, b)] = np.where(first >= MIN_PKTS_PATHLOSS, first, 0.0)
    dir_pairs = {}
    for (a, b), path in paths.items():
        for u, v in zip(path[:-1], path[1:]):
            dir_pairs.setdefault((u, v), []).append((a, b))

    # ---------------- ground-truth noise / link state per link ----------------
    def spans(kind_on, kind_off):
        out = {}
        open_ = {}
        for a in sorted(actions, key=lambda x: x['ts_ns']):
            if a['kind'] == kind_on:
                open_[a['link_id']] = a
            elif a['kind'] == kind_off and a['link_id'] in open_:
                s = open_.pop(a['link_id'])
                out.setdefault(a['link_id'], []).append((s['ts_ns'], a['ts_ns'], s))
        for lid, s in open_.items():
            out.setdefault(lid, []).append((s['ts_ns'], edges[-1] + NS, s))
        return out
    noise_spans, down_spans = spans('noise_on', 'noise_off'), spans('link_down', 'link_up')

    def overlap(sp):
        frac = np.zeros(K)
        jit = np.zeros(K)
        loss = np.zeros(K)
        extra = np.zeros(K)
        for a, b, s in sp:
            ov = np.clip(np.minimum(edges[1:], b) - np.maximum(edges[:-1], a), 0, None) / (INTERVAL_S * NS)
            frac += ov
            on = ov > 0
            jit[on] = s.get('jitter_ms', 0.0)
            loss[on] = s.get('loss_pct', 0.0)
            extra[on] = s.get('extra_delay_ms', 0.0)
        return np.clip(frac, 0, 1), jit, loss, extra

    rows = []
    for d in dirs:
        u, v, lid = d['src'], d['dst'], d['link_id']
        e_intf, i_intf = d['egress_intf'], d['ingress_intf']
        col = {}
        g = pgrp.get(e_intf)
        h = pgrp.get(i_intf)
        nan = np.full(K, np.nan)
        txb = _diff_at(g.ts_ns.values, g.tx_bytes.values, edges) if g is not None else nan
        txp = _diff_at(g.ts_ns.values, g.tx_packets.values, edges) if g is not None else nan
        txd = _diff_at(g.ts_ns.values, g.tx_dropped.values, edges) if g is not None else nan
        rxp = _diff_at(h.ts_ns.values, h.rx_packets.values, edges) if h is not None else nan
        rxd = _diff_at(h.ts_ns.values, h.rx_dropped.values, edges) if h is not None else nan
        col['f_util'] = txb * 8 / (INTERVAL_S * CAPACITY_MBPS * 1e6)
        col['f_tx_mbps'] = txb * 8 / INTERVAL_S / 1e6
        col['f_tx_pps'] = txp / INTERVAL_S
        col['f_rx_pps_far'] = rxp / INTERVAL_S
        col['f_tx_dropped'] = txd
        col['f_rx_dropped_far'] = rxd
        # probes
        ps = pstats.loc[(u, v)] if (u, v) in pstats.index.droplevel(2) else None
        for name, src in (('f_probe_sent', 'sent'), ('f_probe_recv', 'recv'), ('f_delay_mean_ms', 'd_mean'),
                          ('f_delay_min_ms', 'd_min'), ('f_delay_max_ms', 'd_max'), ('f_delay_std_ms', 'd_std'),
                          ('f_delay_p50_ms', 'd_p50'), ('f_probe_reordered', 'reord')):
            arr = np.full(K, np.nan)
            if ps is not None:
                arr[ps.index.values] = ps[src].values
            col[name] = arr
        col['f_probe_sent'] = np.nan_to_num(col['f_probe_sent'])
        col['f_probe_recv'] = np.nan_to_num(col['f_probe_recv'])
        col['f_probe_lost'] = col['f_probe_sent'] - col['f_probe_recv']
        with np.errstate(invalid='ignore', divide='ignore'):
            col['f_probe_loss_ratio'] = np.where(col['f_probe_sent'] > 0, col['f_probe_lost'] / col['f_probe_sent'], np.nan)
        # path loss of flows crossing this direction (2 s window ending at the interval end)
        prs = [p for p in dir_pairs.get((u, v), []) if p in pair_loss]
        if prs:
            L = np.vstack([pair_loss[p] for p in prs])
            W = np.vstack([pair_pk[p] for p in prs])
            with np.errstate(invalid='ignore', divide='ignore'):
                col['f_pathloss_wmean'] = np.where(W.sum(0) > 0, np.nansum(L * W, 0) / W.sum(0), np.nan)
                col['f_pathloss_max'] = np.where(np.isnan(L).all(0), np.nan, np.nanmax(np.where(np.isnan(L), -1, L), 0))
            col['f_path_pairs_active'] = (W > 0).sum(0).astype(float)
            col['f_path_pkts_2s'] = W.sum(0)
        else:
            col['f_pathloss_wmean'] = col['f_pathloss_max'] = nan
            col['f_path_pairs_active'] = col['f_path_pkts_2s'] = np.zeros(K)
        # ---------------- ground truth ----------------
        tq, nq = qgrp.get((e_intf, 'tbf')), qgrp.get((e_intf, 'netem'))
        if tq is not None:
            kk = ((tq.ts_ns.values - t0) // int(INTERVAL_S * NS)).astype(int)
            m = (kk >= 0) & (kk < K)
            bdf = pd.DataFrame({'k': kk[m], 'b': tq.backlog_bytes.values[m], 'q': tq.qlen_packets.values[m]})
            bg = bdf.groupby('k')
            for name, s in (('gt_backlog_mean_bytes', bg.b.mean()), ('gt_backlog_max_bytes', bg.b.max()),
                            ('gt_backlog_min_bytes', bg.b.min()), ('gt_qlen_mean_pkts', bg.q.mean()),
                            ('gt_queue_samples', bg.size())):
                arr = np.full(K, np.nan)
                arr[s.index.values] = s.values
                col[name] = arr
            col['gt_queue_samples'] = np.nan_to_num(col['gt_queue_samples'])
            idx = np.searchsorted(tq.ts_ns.values, edges[1:], side='right') - 1
            col['gt_backlog_snapshot_bytes'] = np.where(idx >= 0, tq.backlog_bytes.values[np.clip(idx, 0, None)], np.nan)
            col['gt_tbf_drops'] = _diff_at(tq.ts_ns.values, tq.drops.values, edges)
        else:
            for name in ('gt_backlog_mean_bytes', 'gt_backlog_max_bytes', 'gt_backlog_min_bytes', 'gt_qlen_mean_pkts',
                         'gt_backlog_snapshot_bytes', 'gt_tbf_drops'):
                col[name] = nan
            col['gt_queue_samples'] = np.zeros(K)
        if nq is not None:
            col['gt_netem_drops_raw'] = _diff_at(nq.ts_ns.values, nq.drops.values, edges)
            sub = np.nan_to_num(col['gt_tbf_drops']) if netem_counts_child else 0.0
            col['gt_noise_drops'] = np.clip(col['gt_netem_drops_raw'] - sub, 0, None)
        else:
            col['gt_netem_drops_raw'] = np.zeros(K)
            col['gt_noise_drops'] = np.zeros(K)
        nf, nj, nl, nx = overlap(noise_spans.get(lid, []))
        col['gt_noise_extra_delay_ms'] = nx
        df_, _, _, _ = overlap(down_spans.get(lid, []))
        col['gt_noise_frac'], col['gt_noise_jitter_ms'], col['gt_noise_loss_pct'] = nf, nj, nl
        col['gt_link_down_frac'] = df_
        col['gt_backlog_mean_frac'] = col['gt_backlog_mean_bytes'] / QUEUE_LIMIT_BYTES
        col['gt_backlog_snapshot_frac'] = col['gt_backlog_snapshot_bytes'] / QUEUE_LIMIT_BYTES
        lab, reason = _labels(col['gt_backlog_mean_frac'], nf, df_)
        col['gt_label'], col['gt_label_reason'] = lab, reason
        col['gt_label_snapshot'], _ = _labels(col['gt_backlog_snapshot_frac'], nf, df_)
        frame = pd.DataFrame(col)
        frame.insert(0, 'meta_k', np.arange(K))
        frame.insert(1, 'meta_t_s', np.arange(K) * INTERVAL_S)
        frame.insert(2, 'meta_warmup', frame.meta_t_s < WARMUP_S)
        for k_, v_ in (('meta_dir_id', d['dir_id']), ('meta_link_id', lid), ('meta_src', u), ('meta_dst', v),
                       ('meta_src_region', region[u]), ('meta_dst_region', region[v]),
                       ('static_capacity_mbps', CAPACITY_MBPS), ('static_prop_delay_ms', links[lid]['delay_ms']),
                       ('static_queue_limit_bytes', QUEUE_LIMIT_BYTES)):
            frame[k_] = v_
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    for k_, v_ in (('meta_run_id', meta['run_id']), ('meta_split', meta['split']), ('meta_topology', cfg['topology']),
                   ('meta_load', cfg['load']), ('meta_noise', cfg['noise']), ('meta_transport', cfg['transport']),
                   ('meta_event', cfg['event']), ('meta_seed', cfg['seed'])):
        out.insert(0, k_, v_)
    out.to_csv(run_dir / 'intervals.csv.gz', index=False, float_format='%.6g', compression='gzip')
    flow_files(run_dir, meta, sched)
    qc = quality(out, meta, sched, actions, ports, queues, flows, send_err)
    write_json(run_dir / 'qc.json', qc)
    (run_dir / 'BUILT').write_text(qc['status'])
    if not quiet:
        print(f"{meta['run_id']}: {len(out)} rows, labels {qc['labels_after_warmup']}, QC {qc['status']} {qc['problems']}")
    return out, qc


def _labels(frac, noise_frac, down_frac):
    K = len(frac)
    high = np.nan_to_num(frac, nan=-1) > LABEL_HIGH
    low = (~np.isnan(frac)) & (frac < LABEL_LOW)
    consec = high.copy()
    for s in range(1, LABEL_CONSEC):
        consec[s:] &= high[:-s]
        consec[:s] = False
    lab = np.full(K, 'uncertain', dtype=object)
    reason = np.full(K, '', dtype=object)
    down = down_frac > 0
    full_noise, no_noise = noise_frac >= 0.999, noise_frac <= 0.0
    lab[consec & ~down] = 'congestion'
    deg = ~consec & ~down & full_noise & low
    nor = ~consec & ~down & no_noise & low
    lab[deg] = 'degradation'
    lab[nor] = 'normal'
    unc = lab == 'uncertain'
    reason[unc & down] = 'link_down'
    reason[unc & ~down & np.isnan(frac)] = 'no_queue_sample'
    reason[unc & ~down & high & ~consec] = 'queue_high_<3_intervals'
    reason[unc & ~down & ~np.isnan(frac) & ~high & ~low] = 'queue_20_to_50pct'
    reason[unc & ~down & low & ~full_noise & ~no_noise] = 'noise_partial_interval'
    return lab, reason


def flow_files(run_dir, meta, sched):
    raw = run_dir / 'raw'
    t0 = meta['t0_ns']
    paths = meta['network']['paths']

    def cat(pattern):
        fs = glob.glob(str(raw / pattern))
        parts = [x for x in (pd.read_csv(f) for f in fs) if len(x)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    plan = pd.DataFrame(sched['tcp_flows'])
    tx, span, sec = cat('tx_tcp_*.csv.gz'), cat('rx_tcp_span_*.csv.gz'), cat('rx_tcp_sec_*.csv.gz')
    if len(plan):
        f = plan[['id', 'src', 'dst', 'cc', 'size', 'start']].rename(columns={'id': 'flow_id', 'start': 'planned_start_s'})
        if len(tx):
            tx = tx.rename(columns={'flow_id': 'flow_id'})
            f = f.merge(tx[['flow_id', 'start_ns', 'end_ns', 'bytes_sent', 'status']], on='flow_id', how='left')
        if len(span):
            f = f.merge(span.rename(columns={'bytes': 'bytes_received'}), on='flow_id', how='left')
        f['status'] = f.get('status', pd.Series(index=f.index, dtype=object)).fillna('not_started')
        f['start_s'] = (f.start_ns - t0) / 1e9
        f['last_rx_s'] = (f.last_ns - t0) / 1e9 if 'last_ns' in f else np.nan
        f['fct_s'] = f.last_rx_s - f.start_s
        f['completed'] = f.get('bytes_received', 0).fillna(0) >= f['size']
        f['path'] = [' '.join(map(str, paths[f'{a}-{b}'])) for a, b in zip(f.src, f.dst)]
        f.drop(columns=[c for c in ('start_ns', 'end_ns', 'first_ns', 'last_ns') if c in f], inplace=True)
        f.to_csv(run_dir / 'flows_tcp.csv.gz', index=False, compression='gzip')
    udp = pd.DataFrame([{'flow_id': u['id'], 'src': u['src'], 'dst': u['dst'], 'kind': u['kind'],
                         'planned_mb': sum((b - a) * r for a, b, r in u['segments']) / 8} for u in sched['udp_flows']])
    useq, utx = cat('rx_udp_seq_*.csv.gz'), cat('tx_udp_sec_*.csv.gz')
    if len(udp):
        if len(utx):
            udp = udp.merge(utx.groupby('flow_id').packets.sum().rename('tx_packets').reset_index(), on='flow_id', how='left')
        if len(useq):
            udp = udp.merge(useq[['flow_id', 'received', 'reordered']], on='flow_id', how='left')
        if 'tx_packets' in udp and 'received' in udp:
            udp['loss_ratio'] = 1 - udp.received.fillna(0) / udp.tx_packets
        udp.to_csv(run_dir / 'flows_udp.csv.gz', index=False, compression='gzip')
    usec = cat('rx_udp_sec_*.csv.gz')
    parts = []
    if len(sec):
        parts.append(sec.assign(proto='tcp', packets=np.nan))
    if len(usec):
        parts.append(usec.assign(proto='udp'))
    if parts:
        s = pd.concat(parts, ignore_index=True)
        s = s[s.sec >= 0]
        cc = {f_['id']: f_['cc'] for f_ in sched['tcp_flows']}
        s['cc'] = s.flow_id.map(cc).fillna('udp')
        s.to_csv(run_dir / 'flow_seconds.csv.gz', index=False, compression='gzip')


def quality(df, meta, sched, actions, ports, queues, flows, send_err):
    probs, warn = [], []
    K = int(round(meta['run_s'] / INTERVAL_S))
    nd = len(meta['network']['directions'])
    post = df[~df.meta_warmup]
    exp_ports = K + 8
    port_cov = ports.groupby('intf').size().min() / exp_ports
    q_ok = float((post.gt_queue_samples >= 3).mean())
    probe_cov = float(((post.f_probe_sent >= PROBE_HZ * INTERVAL_S - 1) & (post.f_probe_recv >= 3)).mean())
    clean = post[(post.gt_label == 'normal')]
    sent = clean.f_probe_sent.sum()
    unexplained = float(clean.f_probe_lost.sum() / sent) if sent else float('nan')
    util_max = float(np.nanmax(df.f_util)) if len(df) else float('nan')
    limit = meta['run_s'] + 0.5   # actions after the run ends are never executed (run is over)
    n_planned = sum((e['start'] <= limit) + (e['end'] <= limit) for e in sched['noise']) + \
        sum((e['start'] <= limit) + (e['end'] <= limit) for e in sched['events'] if e['type'] == 'link_down')
    bad_actions = [a for a in actions if any(rc != 0 for rc in (a['rc'] if isinstance(a['rc'], list) else [a['rc']]))]
    base = clean.groupby('meta_dir_id').apply(lambda g: np.nanmedian(g.f_delay_min_ms - g.static_prop_delay_ms)) \
        if len(clean) and not meta['sandbox'] else pd.Series(dtype=float)
    if port_cov < 0.95:
        probs.append(f'port samples {port_cov:.0%} of expected')
    if q_ok < 0.95:
        probs.append(f'only {q_ok:.0%} of intervals have >=3 queue samples')
    if probe_cov < 0.95:
        warn.append(f'probe coverage {probe_cov:.0%}')
    if not np.isnan(unexplained) and unexplained > 0.02:
        probs.append(f'unexplained probe loss on clean links {unexplained:.2%} (machine overloaded?)')
    elif not np.isnan(unexplained) and unexplained > 0.005:
        warn.append(f'unexplained probe loss on clean links {unexplained:.2%}')
    over = float((df.f_util > 1.10).mean())
    # single readings slightly above 1 are expected: counters are read a few ms off the interval
    # edges and the tbf allows a 15 KB burst; only warn if it is frequent
    if over > 0.001:
        warn.append(f'utilisation above 110% in {over:.2%} of intervals')
    if len(actions) != n_planned:
        probs.append(f'{len(actions)} noise/link actions executed, {n_planned} planned')
    if bad_actions:
        probs.append(f'{len(bad_actions)} noise/link actions failed')
    if len(base) and (base.abs() > 1.5).any():
        warn.append(f'base delay differs from configured on {int((base.abs() > 1.5).sum())} directions')
    status = 'FAIL' if probs else ('WARN' if warn else 'PASS')
    return {'status': status, 'problems': probs, 'warnings': warn,
            'rows': len(df), 'intervals_per_direction': K, 'directions': nd,
            'labels_after_warmup': post.gt_label.value_counts().to_dict(),
            'uncertain_reasons_after_warmup': post[post.gt_label == 'uncertain'].gt_label_reason.value_counts().to_dict(),
            'port_sample_coverage': round(float(port_cov), 4), 'queue_sample_ok_share': round(q_ok, 4),
            'median_tc_call_ms': round(float(queues.call_us.median()) / 1000, 2) if len(queues) else None,
            'probe_coverage': round(probe_cov, 4), 'unexplained_probe_loss_clean': unexplained,
            'max_util': util_max, 'share_util_above_1_10': over, 'noise_actions_logged': len(actions), 'noise_actions_planned': n_planned,
            'failed_actions': len(bad_actions), 'probe_send_errors': int(len(send_err)),
            'base_delay_minus_configured_ms_median_by_dir': {int(k): round(float(v), 3) for k, v in base.items()},
            'target_max_util': sched['target_max_util'], 'realised_max_offered_util': sched['realised_max_offered_util'],
            'realised_max_background_util': sched.get('realised_max_background_util'),
            'load_relative_error': sched.get('load_relative_error'), 'schedule_draw': sched.get('draw')}


if __name__ == '__main__':
    for p in sys.argv[1:]:
        build_run(p)
