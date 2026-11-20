"""
RQ2 evaluation: regional agents (two time-scales) vs one central agent, plus the variant without a fast loop.
Cost metrics come from controller_log.json; quality metrics (reroutes, false reroutes, delay, throughput)
from live_per_run.csv (run live_eval.py first).
  python3 rq2_eval.py   -> ../models/results/rq2_per_run.csv, rq2_tests.csv
"""
import os, json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
LIVE_DIR = Path(os.environ.get('CA_LIVE_DIR', HERE.parent / 'live_runs'))
OUT = HERE.parent / 'models' / 'results'
WARM, IV = 20, 0.25
POL = ('central', 'regional', 'regional_nofast', 'regional_v2')
KEY = ['topology', 'load', 'noise', 'seed']
pd.set_option('display.width', 250)

rows = []
for rd in sorted(Path(p) for p in glob.glob(str(LIVE_DIR / '*__*'))):
    if not (rd / 'DONE').exists():
        continue
    cl = json.loads((rd / 'controller_log.json').read_text())
    if cl.get('policy') not in POL:
        continue
    meta = json.loads((rd / 'meta.json').read_text()); cfg = meta['config']
    st = pd.DataFrame(cl['steps']); st = st[st.k * IV >= WARM]
    run_s = len(cl['steps']) * IV; dur = run_s - WARM; q = cl['rq2']
    mv = [m for m in cl['moves'] if m['t_s'] >= WARM]
    rows.append(dict(run=rd.name, policy=cl['policy'], topology=cfg['topology'], load=cfg['load'], noise=cfg['noise'], seed=cfg['seed'],
                     latency_ms_mean=st.lat_ms.mean(), latency_ms_p95=st.lat_ms.quantile(0.95),
                     compute_ms_mean=(st.lat_ms - st.ctl_rtt_ms).mean(), ctl_rtt_ms=q['ctl_rtt_ms'],
                     cpu_ms_per_interval=st.cpu_ms.mean(), late_ms_p95=st.late_ms.quantile(0.95),
                     longhaul_msgs_per_s=q['msgs_longhaul'] / run_s, local_msgs_per_s=q['msgs_local'] / run_s,
                     flowmods_per_s=cl['n_flowmods'] / dur, remote_flowmods_per_s=q['n_flowmods_remote'] / dur,
                     route_changes_per_min=len(mv) / (dur / 60), maxrss_mb=q['maxrss_mb'], errors=len(cl['errors'])))
R = pd.DataFrame(rows)
if R.empty:
    raise SystemExit('no finished RQ2 runs found')
pr = OUT / 'live_per_run.csv'
QUAL = ['reroutes', 'false_reroute_rate', 'flag_recall_congestion', 'path_delay_mean_ms', 'path_delay_p95_ms', 'tcp_throughput_mbps', 'fct_p95_s']
if pr.exists():
    Q = pd.read_csv(pr)
    R = R.merge(Q[['run'] + [c for c in QUAL if c in Q.columns]], on='run', how='left')
R.to_csv(OUT / 'rq2_per_run.csv', index=False)
print(f'{len(R)} RQ2 runs ({R.groupby("policy").size().to_dict()}), controller errors: {int(R.errors.sum())}')

COST = ['latency_ms_mean', 'latency_ms_p95', 'compute_ms_mean', 'ctl_rtt_ms', 'cpu_ms_per_interval', 'longhaul_msgs_per_s',
        'local_msgs_per_s', 'flowmods_per_s', 'remote_flowmods_per_s', 'route_changes_per_min', 'maxrss_mb']
cols = COST + [c for c in QUAL if c in R.columns]
for topo in sorted(R.topology.unique()):
    print(f'\n=== {topo}: mean per controller ===')
    print(R[R.topology == topo].groupby('policy')[cols].mean().T.round(3).to_string())


def paired(a_pol, b_pol, m):
    a = R[R.policy == a_pol].set_index(KEY)[m]; b = R[R.policy == b_pol].set_index(KEY)[m]
    j = pd.concat([a.rename('a'), b.rename('b')], axis=1, join='inner').dropna(); d = j.a - j.b
    if len(j) < 5 or (d == 0).all():
        return None
    if stats.shapiro(d).pvalue > 0.05:
        p = stats.ttest_rel(j.a, j.b).pvalue; eff = d.mean() / d.std(ddof=1); test = 'paired t'
    else:
        p = stats.wilcoxon(j.a, j.b).pvalue; pos, neg = (d > 0).sum(), (d < 0).sum(); eff = (pos - neg) / max(pos + neg, 1); test = 'Wilcoxon'
    return dict(compare=f'{a_pol} vs {b_pol}', metric=m, n=len(j), mean_a=j.a.mean(), mean_b=j.b.mean(),
                rel_change=(j.a.mean() - j.b.mean()) / j.b.mean() if j.b.mean() else np.nan, test=test, p=p, effect=eff)


T = pd.DataFrame([r for a, b in (('regional', 'central'), ('regional', 'regional_nofast'), ('regional_v2', 'central'), ('regional_v2', 'regional'))
                  for m in ['latency_ms_mean', 'latency_ms_p95', 'cpu_ms_per_interval', 'longhaul_msgs_per_s', 'remote_flowmods_per_s',
                            'reroutes', 'false_reroute_rate', 'path_delay_mean_ms', 'tcp_throughput_mbps']
                  if m in R.columns for r in [paired(a, b, m)] if r])
if len(T):
    o = np.argsort(T.p.values); adj = np.empty(len(T)); mx = 0.0
    for r_, i in enumerate(o):
        mx = max(mx, min(1.0, (len(T) - r_) * T.p.values[i])); adj[i] = mx
    T['p_holm'] = adj; T['significant'] = T.p_holm < 0.05
    print('\n=== paired tests (matched by topology, load, noise, seed; Holm-corrected) ===')
    print(T.round(4).to_string(index=False))
    T.to_csv(OUT / 'rq2_tests.csv', index=False)
print('\nsaved rq2_per_run.csv and rq2_tests.csv in', OUT)
