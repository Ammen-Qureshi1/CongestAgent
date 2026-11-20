"""Verify the online feature code against the offline pipeline on a finished run (no network needed).
python3 check_features.py <run_dir_with_intervals.csv.gz>"""
import sys, pathlib, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'datagen')); sys.path.insert(0, str(HERE))
from window_features import add_features
from features_live import FeatureEngine, F_COLS
def add_long(d):
    d = d.sort_values(['meta_run_id', 'meta_dir_id', 'meta_k']).reset_index(drop=True)
    d['x_portloss'] = ((d.f_tx_pps - d.f_rx_pps_far) / d.f_tx_pps.clip(lower=1)).clip(-0.5, 1)
    d['x_excess_delay'] = d.f_delay_mean_ms - d.static_prop_delay_ms
    g = d.groupby(['meta_run_id', 'meta_dir_id'], sort=False); new = {}
    for W in (20, 40):
        for src, name in [('f_pathloss_wmean', 'pathloss'), ('x_portloss', 'portloss'), ('f_delay_std_ms', 'delaystd'),
                          ('x_excess_delay', 'excessdelay'), ('f_probe_loss_ratio', 'probeloss')]:
            new[f'L{W}_{name}_mean'] = g[src].rolling(W, min_periods=W).mean().reset_index(level=[0, 1], drop=True)
        new[f'L{W}_delaystd_max'] = g['f_delay_std_ms'].rolling(W, min_periods=W).max().reset_index(level=[0, 1], drop=True)
        new[f'L{W}_pathloss_std'] = g['f_pathloss_wmean'].rolling(W, min_periods=W).std().reset_index(level=[0, 1], drop=True)
    return pd.concat([d, pd.DataFrame(new)], axis=1)
d = pd.read_csv(pathlib.Path(sys.argv[1]) / 'intervals.csv.gz')
off = add_long(add_features(d)).sort_values(['meta_k', 'meta_dir_id']).reset_index(drop=True)
dirs = d.drop_duplicates('meta_dir_id').sort_values('meta_dir_id')
eng = FeatureEngine(list(zip(dirs.meta_src, dirs.meta_dst)), dirs.static_prop_delay_ms.values)
rows = []
for k, g in d.sort_values(['meta_k', 'meta_dir_id']).groupby('meta_k'):
    o = eng.step({c: g[c].values for c in F_COLS}); o['meta_k'] = np.full(len(g), k); o['meta_dir_id'] = g.meta_dir_id.values
    rows.append(pd.DataFrame(o))
on = pd.concat(rows, ignore_index=True)
cols = [c for c in on.columns if c in off.columns and c not in ('meta_k', 'meta_dir_id') and not c.startswith('meta')]
m = (~d.sort_values(['meta_k', 'meta_dir_id']).f_delay_mean_ms.isna()).values   # rows not affected by the look-ahead fill
bad = 0
for c in cols:
    a, b = on[c].values.astype(float), off[c].values.astype(float)
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=1e-4, atol=1e-4) | both_nan
    rate = 1 - close.mean()
    if rate > 0.001:
        bad += 1; print(f'MISMATCH {c}: {rate:.2%} of rows differ; example online={a[~close][:3]} offline={b[~close][:3]}')
print(f'checked {len(cols)} features on {len(on)} rows: {len(cols) - bad} match (>=99.9% rows), {bad} differ')
