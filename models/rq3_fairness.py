import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/ca_dataset/dataset')
OUT = 'results'; os.makedirs(OUT, exist_ok=True)
BUSY = 0.8      # a link-second counts as a shared bottleneck if utilisation >= 80%
WARM = 20       # seconds of warm-up dropped

F = pd.read_csv(f'{D}/flows_tcp_all.csv.gz')
S = pd.read_csv(f'{D}/flow_seconds_all.csv.gz'); S = S[S.proto == 'tcp']
cols = ['meta_run_id', 'meta_src', 'meta_dst', 'meta_k', 'f_util', 'meta_load', 'meta_noise', 'meta_topology', 'meta_transport', 'meta_event', 'meta_split']
I = pd.concat([pd.read_csv(f, usecols=cols) for f in sorted(__import__('glob').glob(f'{D}/intervals_*.csv.gz')) if 'features' not in f], ignore_index=True)
I['sec'] = I.meta_k // 4
U = I.groupby(['meta_run_id', 'meta_src', 'meta_dst', 'sec']).f_util.mean().rename('util').reset_index()
RUN = I.drop_duplicates('meta_run_id').set_index('meta_run_id')[['meta_load', 'meta_noise', 'meta_topology', 'meta_transport', 'meta_event', 'meta_split']]
print(f'runs {F.run_id.nunique()} | TCP flows {len(F)} (cubic {int((F.cc=="cubic").sum())}, bbr {int((F.cc=="bbr").sum())})', flush=True)

# --- link-second fairness: flows that were active for the WHOLE second and crossed the same busy link direction ---
F['first_full'] = np.floor(F.start_s) + 1; F['last_full'] = np.floor(F.last_rx_s) - 1
S = S.merge(F[['run_id', 'flow_id', 'path', 'first_full', 'last_full']], on=['run_id', 'flow_id'])
S = S[(S.sec >= S.first_full) & (S.sec <= S.last_full) & (S.sec >= WARM)]
hops = F[['run_id', 'flow_id', 'path']].dropna().copy()
hops['hop'] = hops.path.str.split().apply(lambda p: list(zip(map(int, p[:-1]), map(int, p[1:]))))
hops = hops.explode('hop').dropna(); hops['meta_src'] = hops.hop.str[0]; hops['meta_dst'] = hops.hop.str[1]
L = S.merge(hops[['run_id', 'flow_id', 'meta_src', 'meta_dst']], on=['run_id', 'flow_id'])
L = L.rename(columns={'run_id': 'meta_run_id'}).merge(U, on=['meta_run_id', 'meta_src', 'meta_dst', 'sec'])
L = L[L.util >= BUSY]

K = ['meta_run_id', 'meta_src', 'meta_dst', 'sec']
L['x'] = L.bytes.astype(float); L['x2'] = L.x ** 2
L['xc'] = np.where(L.cc == 'cubic', L.x, 0.0); L['xb'] = np.where(L.cc == 'bbr', L.x, 0.0)
L['nc'] = (L.cc == 'cubic').astype(int); L['nb'] = (L.cc == 'bbr').astype(int)
G = L.groupby(K)[['x', 'x2', 'xc', 'xb', 'nc', 'nb']].sum().reset_index()
G['n'] = G.nc + G.nb
G['jain_all'] = np.where(G.x2 > 0, G.x ** 2 / (G.n * G.x2.where(G.x2 > 0, 1)), np.nan)
both = (G.nc > 0) & (G.nb > 0)
mc = G.xc / G.nc.where(G.nc > 0, 1); mb = G.xb / G.nb.where(G.nb > 0, 1)
den = 2 * (mc ** 2 + mb ** 2)
G['jain_cubic_vs_bbr'] = np.where(both & (den > 0), (mc + mb) ** 2 / den.where(den > 0, 1), np.nan)
G['bbr_over_cubic'] = np.where(both & (mc > 0), mb / mc.where(mc > 0, 1), np.nan)
G['bbr_share'] = np.where(both & (G.x > 0), G.xb / G.x.where(G.x > 0, 1), np.nan)
G = G[G.n >= 2]
R = G.groupby('meta_run_id').agg(jain_all=('jain_all', 'mean'), jain_cubic_vs_bbr=('jain_cubic_vs_bbr', 'mean'),
                                 bbr_over_cubic=('bbr_over_cubic', 'median'), bbr_share=('bbr_share', 'mean'))
R['shared_link_seconds'] = G.groupby('meta_run_id').size()
R = R.join(RUN)

# --- flow-level: goodput of completed flows by size, CUBIC vs BBR ---
F = F.join(RUN, on='run_id')
C = F[F.completed & (F.start_s >= WARM) & (F.fct_s > 0)].copy()
C['goodput_mbps'] = C['size'] * 8 / C.fct_s / 1e6
C['size_bin'] = pd.cut(C['size'], [0, 1e5, 1e6, 1e9], labels=['<100KB', '100KB-1MB', '>1MB'])

def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2: return [float(v.mean()) if len(v) else None, None, None, len(v)]
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [round(float(v.mean()), 4), round(float(v.mean() - h), 4), round(float(v.mean() + h), 4), len(v)]

res = {'settings': dict(busy_util=BUSY, warmup_s=WARM), 'per_run': R.reset_index().to_dict('records'), 'by_condition': {}}
mixed = R[R.meta_transport == 'mixed']
print('\n=== Jain index on busy shared links (mean [95% CI], unit = run) - shortest-path routing, no controller ===')
print(f"{'condition':26s} {'runs':>4s} {'Jain CUBIC-vs-BBR':>26s} {'BBR/CUBIC (median)':>26s} {'Jain all flows':>26s}")
def row(name, sub):
    a, b, c = ci(sub.jain_cubic_vs_bbr), ci(sub.bbr_over_cubic), ci(sub.jain_all)
    res['by_condition'][name] = dict(jain_cubic_vs_bbr=a, bbr_over_cubic=b, jain_all=c)
    f = lambda z: '-' if z[0] is None else (f'{z[0]:.3f} [{z[1]:.3f},{z[2]:.3f}]' if z[1] is not None else f'{z[0]:.3f}')
    print(f'{name:26s} {len(sub):4d} {f(a):>26s} {f(b):>26s} {f(c):>26s}')
row('ALL mixed runs', mixed)
for k in ['meta_load', 'meta_noise', 'meta_topology', 'meta_event']:
    for v, sub in mixed.groupby(k): row(f'{k[5:]}={v}', sub)
cub = R[R.meta_transport == 'cubic']
if len(cub): row('all-CUBIC runs (jain all)', cub)

print('\n=== Flow-level goodput (completed flows, median Mbps) ===')
t = C[C.meta_transport == 'mixed'].groupby(['size_bin', 'cc'], observed=True).goodput_mbps.median().unstack()
t['bbr/cubic'] = t.bbr / t.cubic; print(t.round(3).to_string())
res['goodput_median_by_size'] = t.round(4).reset_index().astype(str).to_dict('records')
fc = F[F.meta_transport == 'mixed'].groupby('cc').completed.mean(); res['completion_rate'] = fc.round(4).to_dict()
print('completion rate:', fc.round(4).to_dict())
# paired run-level test: does BBR get more than CUBIC? (Wilcoxon on log ratio, unit = run)
lr = np.log(mixed.bbr_over_cubic.dropna())
if len(lr) >= 6:
    w = stats.wilcoxon(lr); res['wilcoxon_bbr_vs_cubic'] = dict(n=int(len(lr)), statistic=float(w.statistic), p=float(w.pvalue))
    print(f'Wilcoxon (BBR vs CUBIC throughput on shared links, n={len(lr)} runs): p = {w.pvalue:.2e}')
R.to_csv(f'{OUT}/rq3_per_run.csv'); json.dump(res, open(f'{OUT}/rq3_fairness.json', 'w'), indent=1, default=str)
print('\nsaved results/rq3_fairness.json and results/rq3_per_run.csv')
