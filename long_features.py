"""
long_features.py - the 14 long-window features (5 s and 10 s) that the CongestAgent model uses on top of the
51 base features in the *_features.csv.gz files. Usage:

    import pandas as pd
    from long_features import add_long, usable
    train = usable(add_long(pd.read_csv("intervals_train_features.csv.gz", low_memory=False)))

add_long() sorts by run, link direction and interval, and rolls only inside one run + one link direction,
so nothing leaks between runs. Feature columns = those starting with f_ p_ w_ nb_ static_ (not w_complete)
plus the new L20_* and L40_* columns.
Utilisation gate used by our controller: call a link "congested" only if w_util_mean >= 0.541
(chosen on the train split; see live_rules.json).
"""
import numpy as np, pandas as pd

KEY = ['meta_run_id', 'meta_dir_id']
CL = ['normal', 'degradation', 'congestion']


def add_long(d):
    d = d.sort_values(['meta_run_id', 'meta_dir_id', 'meta_k']).reset_index(drop=True)
    d['x_portloss'] = ((d.f_tx_pps - d.f_rx_pps_far) / d.f_tx_pps.clip(lower=1)).clip(-0.5, 1)
    d['x_excess_delay'] = d.f_delay_mean_ms - d.static_prop_delay_ms
    g = d.groupby(KEY, sort=False); new = {}
    for W in (20, 40):
        for src, name in [('f_pathloss_wmean', 'pathloss'), ('x_portloss', 'portloss'), ('f_delay_std_ms', 'delaystd'),
                          ('x_excess_delay', 'excessdelay'), ('f_probe_loss_ratio', 'probeloss')]:
            new[f'L{W}_{name}_mean'] = g[src].rolling(W, min_periods=W).mean().reset_index(level=[0, 1], drop=True)
        new[f'L{W}_delaystd_max'] = g['f_delay_std_ms'].rolling(W, min_periods=W).max().reset_index(level=[0, 1], drop=True)
        new[f'L{W}_pathloss_std'] = g['f_pathloss_wmean'].rolling(W, min_periods=W).std().reset_index(level=[0, 1], drop=True)
    return pd.concat([d, pd.DataFrame(new)], axis=1)


def usable(d):
    """rows to train/test on: after warm-up, full 2 s window, certain label"""
    return d[(~d.meta_warmup) & d.w_complete & d.gt_label.isin(CL)].reset_index(drop=True)
