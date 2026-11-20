"""
Extra live-run analysis (after live_eval.py):
  A) where the reroutes of every controller were triggered (label of the trigger link) and paired tests on the
     RQ1 metrics that live_eval does not test: number / share of reroutes on degraded links, degradation flag rate,
     congestion recall
  B) RQ3 under control: CUBIC-vs-BBR fairness (Jain index on busy shared links) per controller, from the same live runs.
     The route in force for every second is rebuilt from the controller's move log, so rerouted flows are counted on
     the links they really used.
  python3 live_extra.py   -> ../models/results/live_extra_tests.csv, live_rq3_per_run.csv
"""
import os, json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
LIVE_DIR = Path(os.environ.get('CA_LIVE_DIR', HERE.parent / 'live_runs'))
OUT = HERE.parent / 'models' / 'results'
WARM, BUSY = 20, 0.8
KEY = ['load', 'noise', 'seed']
pd.set_option('display.width', 250)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); mx = 0.0
    for r, i in enumerate(o):
        mx = max(mx, min(1.0, (len(p) - r) * p[i])); adj[i] = mx
    return adj


def paired(R, metric, base, model='model'):
    a = R[R.policy == model].set_index(KEY)[metric]; b = R[R.policy == base].set_index(KEY)[metric]
    j = pd.concat([a.rename('m'), b.rename('b')], axis=1, join='inner').dropna(); d = j.m - j.b
    if len(j) < 5 or (d == 0).all():
        return None
    if stats.shapiro(d).pvalue > 0.05:
        p = stats.ttest_rel(j.m, j.b).pvalue; eff = d.mean() / d.std(ddof=1); test = 'paired t'
    else:
        p = stats.wilcoxon(j.m, j.b).pvalue; pos, neg = (d > 0).sum(), (d < 0).sum(); eff = (pos - neg) / max(pos + neg, 1); test = 'Wilcoxon'
    return dict(baseline=base, metric=metric, n=len(j), model_mean=j.m.mean(), base_mean=j.b.mean(), test=test, p=p, effect=eff)


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan, np.nan, np.nan)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return (v.mean(), v.mean() - h, v.mean() + h)


# ---------------------------------------------------------------- A) where the reroutes happened
R = pd.read_csv(OUT / 'live_per_run.csv')
print(f'{len(R)} live runs')
print('\n=== A1) label of the link that triggered each reroute (mean count per run) ===')
cols = ['reroutes', 'reroutes_on_congestion', 'reroutes_on_degradation', 'reroutes_on_normal', 'reroutes_on_uncertain']
print(R.groupby('policy')[cols].mean().round(1).T.to_string())
print('\nshare of reroutes by trigger label (sum over runs):')
S = R.groupby('policy')[cols].sum()
print((S[cols[1:]].div(S.reroutes, axis=0)).round(3).T.to_string())

tests = []
for base in ('none', 'delay', 'delayloss', 'delayutil'):
    for m in ('reroutes_on_degradation', 'false_reroute_rate', 'flag_rate_degradation', 'flag_recall_congestion'):
        if base == 'none' and m != 'flag_recall_congestion':
            continue
        r = paired(R, m, base)
        if r:
            tests.append(r)

# ---------------------------------------------------------------- B) RQ3 under control
def run_fairness(rd):
    meta = json.loads((rd / 'meta.json').read_text()); cl = json.loads((rd / 'controller_log.json').read_text())
    cfg = meta['config']
    F = pd.read_csv(rd / 'flows_tcp.csv.gz')
    S = pd.read_csv(rd / 'flow_seconds.csv.gz'); S = S[S.proto == 'tcp']
    F['first_full'] = np.floor(F.start_s) + 1; F['last_full'] = np.floor(F.last_rx_s) - 1
    S = S.merge(F[['flow_id', 'src', 'dst', 'first_full', 'last_full']], on='flow_id')
    S = S[(S.sec >= S.first_full) & (S.sec <= S.last_full) & (S.sec >= WARM)]
    if S.empty:
        return None
    # route in force at the middle of every second, per source-destination pair
    tl = {tuple(map(int, k.split('-'))): [(0.0, v)] for k, v in meta['network']['paths'].items()}
    for m in cl['moves']:
        tl[tuple(m['pair'])].append((m['t_s'], m['new']))
    parts = []
    for (s_, d_), g in S.groupby(['src', 'dst']):
        ev = sorted(tl[(s_, d_)], key=lambda x: x[0]); T = np.array([e[0] for e in ev])
        idx = np.searchsorted(T, g.sec.values + 0.5, side='right') - 1
        g = g.assign(pidx=idx)
        hops = pd.DataFrame([(s_, d_, i, a, b) for i, (_, path) in enumerate(ev) for a, b in zip(path[:-1], path[1:])],
                            columns=['src', 'dst', 'pidx', 'hs', 'hd'])
        parts.append(g.merge(hops, on=['src', 'dst', 'pidx']))
    L = pd.concat(parts, ignore_index=True)
    iv = pd.read_csv(rd / 'intervals.csv.gz', usecols=['meta_src', 'meta_dst', 'meta_k', 'f_util'])
    iv['sec'] = iv.meta_k // 4
    U = iv.groupby(['meta_src', 'meta_dst', 'sec']).f_util.mean().rename('util').reset_index().rename(columns={'meta_src': 'hs', 'meta_dst': 'hd'})
    L = L.merge(U, on=['hs', 'hd', 'sec']); L = L[L.util >= BUSY]
    if L.empty:
        return None
    L['x'] = L.bytes.astype(float); L['x2'] = L.x ** 2
    L['xc'] = np.where(L.cc == 'cubic', L.x, 0.0); L['xb'] = np.where(L.cc == 'bbr', L.x, 0.0)
    L['nc'] = (L.cc == 'cubic').astype(int); L['nb'] = (L.cc == 'bbr').astype(int)
    G = L.groupby(['hs', 'hd', 'sec'])[['x', 'x2', 'xc', 'xb', 'nc', 'nb']].sum().reset_index()
    G = G[(G.nc > 0) & (G.nb > 0)]
    if G.empty:
        return None
    mc = G.xc / G.nc; mb = G.xb / G.nb; den = 2 * (mc ** 2 + mb ** 2)
    jain = np.where(den > 0, (mc + mb) ** 2 / den.where(den > 0, 1), np.nan)
    ratio = np.where(mc > 0, mb / mc.where(mc > 0, 1), np.nan)
    return dict(run=rd.name, policy=meta['policy'], load=cfg['load'], noise=cfg['noise'], seed=cfg['seed'],
                jain_cubic_vs_bbr=float(np.nanmean(jain)), bbr_over_cubic=float(np.nanmedian(ratio)), shared_link_seconds=int(len(G)))


rows = []
for rd in sorted(Path(p) for p in glob.glob(str(LIVE_DIR / '*__*'))):
    if (rd / 'DONE').exists() and (rd / 'intervals.csv.gz').exists():
        try:
            r = run_fairness(rd)
            if r:
                rows.append(r)
        except Exception as e:
            print('skip', rd.name, type(e).__name__, e)
Q = pd.DataFrame(rows)
Q.to_csv(OUT / 'live_rq3_per_run.csv', index=False)
print('\n=== B) RQ3 under control: CUBIC-vs-BBR fairness on busy shared links (mean [95% CI], unit = run) ===')
print(f"{'controller':12s} {'runs':>4s} {'Jain CUBIC-vs-BBR':>24s} {'BBR/CUBIC (median)':>24s} {'shared link-seconds':>20s}")
for pol in ('none', 'delay', 'delayloss', 'delayutil', 'model'):
    q = Q[Q.policy == pol] if len(Q) else Q
    if not len(q):
        continue
    a, b = ci(q.jain_cubic_vs_bbr), ci(q.bbr_over_cubic)
    print(f'{pol:12s} {len(q):4d} {a[0]:8.3f} [{a[1]:.3f},{a[2]:.3f}] {b[0]:8.3f} [{b[1]:.3f},{b[2]:.3f}] {int(q.shared_link_seconds.mean()):20d}')
if len(Q):
    for base in ('none', 'delay', 'delayloss', 'delayutil'):
        r = paired(Q, 'jain_cubic_vs_bbr', base)
        if r:
            r['metric'] = 'rq3_jain_cubic_vs_bbr'; tests.append(r)
T = pd.DataFrame(tests)
if len(T):
    T['p_holm'] = holm(T.p.values); T['significant'] = T.p_holm < 0.05
    print('\n=== paired tests: model vs baselines (Holm over these tests only) ===')
    print(T.round(4).to_string(index=False))
    T.to_csv(OUT / 'live_extra_tests.csv', index=False)
print('\nsaved live_extra_tests.csv and live_rq3_per_run.csv in', OUT)
