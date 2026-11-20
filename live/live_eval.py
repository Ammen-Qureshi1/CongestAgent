"""
RQ1 (and RQ2 cost) results from the live runs.  python3 live_eval.py   -> results/live_*.csv/json
Per run: reroutes, FALSE reroutes (trigger link truly had non-congestive degradation), unnecessary reroutes
(trigger link not truly congested), route changes/min, TCP throughput and flow completion, path delay
(mean, p95), UDP loss, controller decision time and OpenFlow flow-mods. Then paired tests: model vs each baseline.
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
LIVE_DIR = Path(os.environ.get('CA_LIVE_DIR', HERE.parent / 'live_runs'))
OUT = HERE.parent / 'models' / 'results'; OUT.mkdir(parents=True, exist_ok=True)
WARM, IV = 20, 0.25


def wq(x, w, q):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]; c = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(c, q)]) if len(x) else np.nan


def one(rd):
    meta = json.loads((rd / 'meta.json').read_text()); cl = json.loads((rd / 'controller_log.json').read_text())
    cfg = meta['config']; run_s = meta['run_s']
    iv = pd.read_csv(rd / 'intervals.csv.gz')
    lab = iv.pivot(index='meta_k', columns='meta_dir_id', values='gt_label')
    dly = iv.pivot(index='meta_k', columns='meta_dir_id', values='f_delay_mean_ms')
    dirs = {(d['src'], d['dst']): d['dir_id'] for d in meta['network']['directions']}
    r = dict(run=rd.name, policy=meta['policy'], load=cfg['load'], noise=cfg['noise'], seed=cfg['seed'])
    mv = pd.DataFrame(cl['moves'])
    rr = mv[mv.kind == 'reroute'] if len(mv) else mv
    labs = [lab.at[m.k, m.trigger_dir] for m in rr.itertuples()] if len(rr) else []
    r['reroutes'] = len(rr); r['route_changes_per_min'] = (len(mv) / ((run_s - WARM) / 60)) if len(mv) else 0.0
    for L in ('congestion', 'degradation', 'normal', 'uncertain'):
        r[f'reroutes_on_{L}'] = int(sum(x == L for x in labs))
    r['false_reroute_rate'] = r['reroutes_on_degradation'] / len(rr) if len(rr) else np.nan       # definition used in the paper
    r['unnecessary_reroute_rate'] = 1 - r['reroutes_on_congestion'] / len(rr) if len(rr) else np.nan
    # online detection quality (what the controller flagged vs the truth), after warm-up
    z = np.load(rd / 'controller_flags.npz'); fl = z['flags'][int(WARM / IV):]
    L = lab.values[int(WARM / IV):]
    for name, m in (('flag_recall_congestion', L == 'congestion'), ('flag_rate_degradation', L == 'degradation'),
                    ('flag_rate_normal', L == 'normal')):
        r[name] = float(fl[m].mean()) if m.any() else np.nan
    # traffic
    f = pd.read_csv(rd / 'flows_tcp.csv.gz')
    f = f[(f.start_s >= WARM) & (f.start_s <= run_s - 10)]
    r['tcp_completion'] = float(f.completed.mean())
    fct = np.where(f.completed, f.fct_s, run_s - f.start_s)
    r['fct_median_s'] = float(np.median(fct)); r['fct_p95_s'] = float(np.percentile(fct, 95))
    sec = pd.read_csv(rd / 'flow_seconds.csv.gz'); tcp = sec[(sec.proto == 'tcp') & (sec.sec >= WARM) & (sec.sec < run_s)]
    r['tcp_throughput_mbps'] = float(tcp.bytes.sum() * 8 / (run_s - WARM) / 1e6)
    u = pd.read_csv(rd / 'flows_udp.csv.gz') if (rd / 'flows_udp.csv.gz').exists() else pd.DataFrame()
    r['udp_loss'] = float(u.loss_ratio.mean()) if 'loss_ratio' in u else np.nan
    # path delay seen by traffic: route in force each second x link delays, weighted by bytes sent that second
    paths = {tuple(map(int, k.split('-'))): v for k, v in meta['network']['paths'].items()}
    timeline = {p: [(0.0, v)] for p, v in paths.items()}
    for m in cl['moves']:
        timeline[tuple(m['pair'])].append((m['t_s'], m['new']))
    allf = pd.read_csv(rd / 'flows_tcp.csv.gz')[['flow_id', 'src', 'dst']]
    t = tcp.merge(allf, on='flow_id').groupby(['src', 'dst', 'sec']).bytes.sum().reset_index()
    sec_delay = dly.groupby(dly.index // 4).mean()
    xs, ws = [], []
    for row in t.itertuples():
        path = [v for ts, v in timeline[(row.src, row.dst)] if ts <= row.sec + 0.5][-1]
        if row.sec not in sec_delay.index:
            continue
        dsum = sum(sec_delay.at[row.sec, dirs[(a, b)]] for a, b in zip(path[:-1], path[1:]))
        if np.isfinite(dsum):
            xs.append(dsum); ws.append(row.bytes)
    r['path_delay_mean_ms'] = float(np.average(xs, weights=ws)) if xs else np.nan
    r['path_delay_p95_ms'] = wq(xs, ws, 0.95) if xs else np.nan
    st = pd.DataFrame(cl['steps'])
    dec = st.measure_ms + st.detect_ms
    r['decision_ms_mean'] = float(dec.mean()); r['decision_ms_p95'] = float(dec.quantile(0.95))
    r['install_ms_per_move'] = float(mv.install_ms.mean()) if len(mv) else np.nan
    r['flowmods_per_s'] = cl['n_flowmods'] / (run_s - WARM)
    r['controller_errors'] = len(cl['errors'])
    return r


def main():
    rows = []
    for rd in sorted(Path(p) for p in glob.glob(str(LIVE_DIR / '*__*'))):
        if (rd / 'DONE').exists() and (rd / 'intervals.csv.gz').exists():
            try:
                rows.append(one(rd))
            except Exception as e:
                print('skip', rd.name, type(e).__name__, e)
    R = pd.DataFrame(rows); R.to_csv(OUT / 'live_per_run.csv', index=False)
    print(f'{len(R)} live runs')
    M = ['reroutes', 'false_reroute_rate', 'unnecessary_reroute_rate', 'route_changes_per_min', 'flag_recall_congestion',
         'flag_rate_degradation', 'tcp_throughput_mbps', 'tcp_completion', 'fct_median_s', 'fct_p95_s',
         'path_delay_mean_ms', 'path_delay_p95_ms', 'udp_loss', 'decision_ms_mean', 'decision_ms_p95', 'flowmods_per_s']
    pd.set_option('display.width', 250)
    print('\n=== mean per controller (all configurations) ===')
    print(R.groupby('policy')[M].mean().round(3).T.to_string())
    print('\n=== false reroute rate by noise ===')
    print(R.pivot_table(index='noise', columns='policy', values='false_reroute_rate').round(3).to_string())
    print('\n=== reroutes by noise ===')
    print(R.pivot_table(index='noise', columns='policy', values='reroutes').round(1).to_string())
    # paired tests: model vs each baseline, matched by (load, noise, seed); Holm over all tests
    res = []
    key = ['load', 'noise', 'seed']
    R = R[R.policy.isin(('none', 'delay', 'delayloss', 'delayutil', 'model', 'model_robust'))]
    pairs = [('model', b) for b in ('none', 'delay', 'delayloss', 'delayutil')]
    if (R.policy == 'model_robust').any():
        pairs += [('model_robust', b) for b in ('none', 'delay', 'delayloss', 'delayutil', 'model')]
    for mp, base in pairs:
        if not (R.policy == base).any():
            continue
        a = R[R.policy == mp].set_index(key); b = R[R.policy == base].set_index(key)
        j = a.join(b, lsuffix='_m', rsuffix='_b', how='inner')
        for m in ('reroutes', 'false_reroute_rate', 'unnecessary_reroute_rate', 'tcp_throughput_mbps', 'path_delay_mean_ms', 'path_delay_p95_ms', 'fct_p95_s'):
            x, y = j[f'{m}_m'], j[f'{m}_b']; ok = x.notna() & y.notna(); dlt = (x - y)[ok]
            if ok.sum() < 5 or (dlt == 0).all():
                continue
            normal = stats.shapiro(dlt).pvalue > 0.05 if ok.sum() >= 3 else False
            if normal:
                p = stats.ttest_rel(x[ok], y[ok]).pvalue; eff = dlt.mean() / dlt.std(ddof=1); test = 'paired t'
            else:
                w = stats.wilcoxon(x[ok], y[ok]); p = w.pvalue
                pos, neg = (dlt > 0).sum(), (dlt < 0).sum(); eff = (pos - neg) / max(pos + neg, 1); test = 'Wilcoxon'
            res.append(dict(controller=mp, baseline=base, metric=m, n=int(ok.sum()), model_mean=x[ok].mean(), base_mean=y[ok].mean(),
                            test=test, p=p, effect=eff))
    T = pd.DataFrame(res)
    if len(T):
        o = np.argsort(T.p.values); mmax = 0; adj = np.empty(len(T))
        for r_, i in enumerate(o):
            mmax = max(mmax, min(1, (len(T) - r_) * T.p.values[i])); adj[i] = mmax
        T['p_holm'] = adj; T['significant'] = T.p_holm < 0.05
        print('\n=== paired tests: model vs baselines (Holm-corrected) ===')
        print(T.round(4).to_string(index=False))
        T.to_csv(OUT / 'live_tests.csv', index=False)
    print(f'\nsaved {OUT}/live_per_run.csv and live_tests.csv')


if __name__ == '__main__':
    main()
