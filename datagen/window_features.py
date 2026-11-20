"""
Preprocessing and engineered features, computed from the raw
interval rows (f_* columns only, never gt_*). All features are CAUSAL: the value
at interval k uses intervals k-7..k (one 2 s window of 8 intervals) of the same
run, so they can be computed online by a controller.

  python3 window_features.py dataset/intervals_train.csv.gz [more files...]
  -> writes <name>_features.csv.gz next to each input

Preprocessing
  * a single missing delay sample is filled from the previous interval; longer gaps
    leave NaN and the row is flagged (w_complete = False)
  * EMA (alpha 0.3) of delay and loss, raw values kept
Features
  * delay level, first and second change, linear trend (slope) in the window
  * delay variance across the window and mean within-interval jitter (probe SD)
  * queueing-delay estimate: delay above the lowest delay seen so far on that direction
  * utilisation mean/max, and the CORRELATION between utilisation and delay in the window
    (strong under queueing, weak under jitter)
  * loss: mean path loss and probe loss; loss when utilisation is high (>=0.8) vs low (<0.5);
    correlation between utilisation and loss ("does loss rise only when the link is busy?")
  * spatial view: the same window features averaged over neighbouring link directions
    (directions that share a node), in the spirit of [3] and [4]
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

W = 8
ALPHA = 0.3


def _per_dir(g):
    g = g.sort_values('meta_k').copy()
    d = g['f_delay_mean_ms']
    miss = d.isna()
    single = miss & ~miss.shift(1, fill_value=False) & ~miss.shift(-1, fill_value=False)
    d = d.where(~single, d.ffill())
    g['p_delay_ms'] = d
    loss = g['f_pathloss_wmean'].fillna(0.0)
    util = g['f_util']
    g['p_delay_ema'] = d.ewm(alpha=ALPHA, adjust=False).mean()
    g['p_loss_ema'] = loss.ewm(alpha=ALPHA, adjust=False).mean()
    r = d.rolling(W, min_periods=W)
    g['w_delay_level'] = r.mean()
    g['w_delay_d1'] = d.diff()
    g['w_delay_d2'] = d.diff().diff()
    x = np.arange(W) - (W - 1) / 2
    g['w_delay_trend'] = d.rolling(W, min_periods=W).apply(lambda v: float(np.dot(x, v - v.mean()) / np.dot(x, x)), raw=True)
    g['w_delay_var'] = r.var(ddof=0)
    g['w_jitter_mean'] = g['f_delay_std_ms'].rolling(W, min_periods=W).mean()
    base = g['f_delay_min_ms'].cummin()
    g['w_queue_delay_est'] = d - base
    g['w_util_mean'] = util.rolling(W, min_periods=W).mean()
    g['w_util_max'] = util.rolling(W, min_periods=W).max()
    g['w_corr_util_delay'] = util.rolling(W, min_periods=W).corr(d).fillna(0.0)
    g['w_loss_mean'] = loss.rolling(W, min_periods=W).mean()
    g['w_probe_loss'] = g['f_probe_loss_ratio'].rolling(W, min_periods=W).mean()
    hi = loss.where(util >= 0.8)
    lo = loss.where(util < 0.5)
    g['w_loss_when_util_high'] = hi.rolling(W, min_periods=1).mean()
    g['w_loss_when_util_low'] = lo.rolling(W, min_periods=1).mean()
    g['w_corr_util_loss'] = util.rolling(W, min_periods=W).corr(loss).fillna(0.0)
    g['w_complete'] = d.rolling(W, min_periods=W).count().eq(W) & util.rolling(W, min_periods=W).count().eq(W)
    return g


WIN = ['w_delay_level', 'w_delay_trend', 'w_delay_var', 'w_jitter_mean', 'w_queue_delay_est', 'w_util_mean',
       'w_corr_util_delay', 'w_loss_mean', 'w_probe_loss', 'w_corr_util_loss']


def add_features(df):
    # explicit loop (behaves the same on every pandas version)
    out = pd.concat([_per_dir(g) for _, g in df.groupby(['meta_run_id', 'meta_dir_id'], sort=False)], ignore_index=True)
    # neighbours: directions sharing a node with this direction (same run, same interval)
    keys = out[['meta_run_id', 'meta_dir_id', 'meta_src', 'meta_dst']].drop_duplicates()
    nb_rows = []
    for run, kk in keys.groupby('meta_run_id'):
        for r in kk.itertuples():
            for o in kk.itertuples():
                if o.meta_dir_id != r.meta_dir_id and {o.meta_src, o.meta_dst} & {r.meta_src, r.meta_dst} \
                        and {o.meta_src, o.meta_dst} != {r.meta_src, r.meta_dst}:
                    nb_rows.append((run, r.meta_dir_id, o.meta_dir_id))
    nb = pd.DataFrame(nb_rows, columns=['meta_run_id', 'meta_dir_id', 'nb_dir'])
    sub = out[['meta_run_id', 'meta_dir_id', 'meta_k'] + WIN].rename(columns={'meta_dir_id': 'nb_dir'})
    m = nb.merge(sub, on=['meta_run_id', 'nb_dir'])
    agg = m.groupby(['meta_run_id', 'meta_dir_id', 'meta_k'])[WIN].mean().add_prefix('nb_').reset_index()
    out = out.merge(agg, on=['meta_run_id', 'meta_dir_id', 'meta_k'], how='left')
    return out


def main():
    for f in [x for x in sys.argv[1:] if not x.endswith('_features.csv.gz')]:
        df = pd.read_csv(f)
        out = add_features(df)
        dest = Path(f).with_name(Path(f).name.replace('.csv.gz', '_features.csv.gz'))
        out.to_csv(dest, index=False, compression='gzip', float_format='%.6g')
        usable = out[(~out.meta_warmup) & (out.gt_label != 'uncertain') & out.w_complete]
        print(f'{dest.name}: {len(out)} rows, {len(usable)} usable for training/testing '
              f'(after warm-up, certain label, full window); labels {usable.gt_label.value_counts().to_dict()}')


if __name__ == '__main__':
    main()
