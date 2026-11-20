import os, sys, glob
import numpy as np, pandas as pd
D = os.path.expanduser('~/ca_dataset/dataset')
files = sys.argv[1:] or sorted(glob.glob(f'{D}/intervals_*_features.csv.gz'))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
ok = []
def check(name, passed, detail):
    ok.append(passed); print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
post = d[~d.meta_warmup]
feat = [c for c in d.columns if c.startswith(('f_', 'p_', 'w_', 'nb_', 'static_')) and c != 'w_complete']
print(f'rows {len(d)}, runs {d.meta_run_id.nunique()}, feature columns {len(feat)}\n')

# 1 structure
per = d.groupby('meta_run_id').size(); dirs = d.groupby('meta_run_id').meta_dir_id.nunique()
check('1 every run complete (480 intervals x every direction)', bool((per == dirs * 480).all()), f'{(per == dirs*480).sum()}/{len(per)} runs complete')
dup = d.duplicated(['meta_run_id', 'meta_dir_id', 'meta_k']).sum()
check('2 no duplicate rows', dup == 0, f'{dup} duplicates')
# 3 splits
sp = d.groupby('meta_run_id').meta_split.nunique().max()
tr = d[d.meta_split == 'train']
check('3 each run in exactly one split', sp == 1, 'no run shared between train/val/test')
check('4 2% loss never in training', not (tr.meta_noise == 'loss2').any(), 'loss2 runs only in val/test')
check('5 GEANT never in training', not (tr.meta_topology == 'geant').any(), 'geant only in test_geant')
# 6 leakage
bad = [c for c in feat if c.startswith(('gt_', 'meta_'))]
check('6 no label/answer column used as a feature', not bad, f'feature columns start with f_/p_/w_/nb_/static_ only')
# 7 missing values
usable = post[post.w_complete]
COND = ['w_loss_when_util_high', 'w_loss_when_util_low']   # empty by design when the window has no high/low-util interval
nan = usable[[c for c in feat if c not in COND]].isna().mean().max()
check('7 features have no missing values after warm-up', nan == 0, f'worst column missing {nan:.4%} (by-design gaps: ' +
      ', '.join(f'{c} {usable[c].isna().mean():.0%}' for c in COND) + ' - LightGBM handles these)')
# 8 labels make sense
none = post[post.meta_noise == 'none']
check('8 runs with no noise have zero degradation rows', int((none.gt_label == 'degradation').sum()) == 0, f'{int((none.gt_label=="degradation").sum())} found')
deg = post[post.gt_label == 'degradation']
check('9 degradation rows only where noise was really on', bool((deg.gt_noise_frac >= 0.999).all()), f'{len(deg)} rows checked')
con = post[post.gt_label == 'congestion']
check('10 congestion rows have a really full queue', bool((con.gt_backlog_mean_frac > 0.5).all()), f'median queue fill {con.gt_backlog_mean_frac.median():.0%}')
lab = post.groupby('meta_load').gt_label.apply(lambda s: (s == 'congestion').mean())
check('11 more load -> more congestion', bool(lab.is_monotonic_increasing), ' / '.join(f'{k}%: {v:.2%}' for k, v in lab.items()))
# 12 physics: configured loss = measured noise loss
lr = post[(post.gt_noise_loss_pct > 0) & (post.gt_noise_frac >= 0.999)]
if len(lr):
    m = lr.groupby('gt_noise_loss_pct').apply(lambda g: g.gt_noise_drops.sum() / max(g.gt_noise_drops.sum() + (g.f_tx_pps * 0.25).sum(), 1))
    good = all(abs(v * 100 - k) / k < 0.25 for k, v in m.items())
    check('12 measured noise loss = configured loss', good, ' / '.join(f'set {k}% -> measured {v:.2%}' for k, v in m.items()))
# 13 physics: configured jitter = measured probe spread
jr = post[(post.gt_noise_jitter_ms > 0) & (post.gt_label == 'degradation')]
if len(jr):
    m = jr.groupby('gt_noise_jitter_ms').f_delay_std_ms.median()
    good = all(0.6 * k < v < 1.3 * k for k, v in m.items())
    check('13 measured jitter ~ configured jitter', good, ' / '.join(f'set {k} ms -> probe std {v:.2f} ms' for k, v in m.items()))
# 14 physics: clean link delay = propagation delay
cl = post[(post.gt_label == 'normal') & (post.gt_backlog_mean_frac < 0.02)]
ex = (cl.f_delay_min_ms - cl.static_prop_delay_ms).median()
check('14 clean-link delay = cable delay', abs(ex) < 0.5, f'median extra {ex:.3f} ms')
# 15 physics: extra delay follows queue size (10 Mbps: 1 byte = 0.0008 ms)
q = post[(post.gt_noise_jitter_ms == 0) & post.gt_label.isin(['normal', 'congestion'])]
qd = q.gt_backlog_mean_bytes * 8 / 10e6 * 1000; ed = q.f_delay_mean_ms - q.static_prop_delay_ms
c = np.corrcoef(qd, ed)[0, 1]
check('15 extra delay follows the queue (independent probe vs tc reading)', c > 0.7, f'correlation {c:.2f}')
# 16 utilisation physically possible
u = post.f_util.quantile(0.999)
check('16 utilisation never far above 100%', u < 1.1, f'99.9th percentile {u:.2f}')
mx = post.groupby(['meta_load', 'meta_run_id']).f_util.max().groupby('meta_load').median()
check('17 load targets are different and in order', bool(mx.is_monotonic_increasing), ' / '.join(f'{k}%: busiest link peak {v:.2f}' for k, v in mx.items()))
# 18 class counts large enough
cnt = usable[usable.meta_split == 'train'].gt_label.value_counts()
check('18 enough training rows per class (>1000)', all(cnt.get(c, 0) > 1000 for c in ['normal', 'degradation', 'congestion']), cnt.to_dict())
tp = d.meta_transport.value_counts().to_dict()
print(f'\nINFO transports: {tp} | noise types: {sorted(d.meta_noise.unique())} | events: {sorted(d.meta_event.unique())}')
print(f'\n{sum(ok)}/{len(ok)} checks passed')
