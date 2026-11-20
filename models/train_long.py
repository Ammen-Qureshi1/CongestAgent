"""Add LONG-window (5 s and 10 s) loss/jitter features and see if normal-vs-degradation improves.
Run: python3 train_long.py"""
import os, sys, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, confusion_matrix
D = os.path.expanduser('~/ca_dataset/dataset'); OUT = 'results'; os.makedirs(OUT, exist_ok=True)
CL = ['normal', 'degradation', 'congestion']
WINS = (20, 40)   # intervals of 250 ms = 5 s, 10 s

def add_long(d):
    d = d.sort_values(['meta_run_id', 'meta_dir_id', 'meta_k']).reset_index(drop=True)
    d['x_portloss'] = ((d.f_tx_pps - d.f_rx_pps_far) / d.f_tx_pps.clip(lower=1)).clip(-0.5, 1)
    d['x_excess_delay'] = d.f_delay_mean_ms - d.static_prop_delay_ms
    g = d.groupby(['meta_run_id', 'meta_dir_id'], sort=False)
    new = {}
    for W in WINS:
        for src, name in [('f_pathloss_wmean', 'pathloss'), ('x_portloss', 'portloss'), ('f_delay_std_ms', 'delaystd'),
                          ('x_excess_delay', 'excessdelay'), ('f_probe_loss_ratio', 'probeloss')]:
            r = g[src].rolling(W, min_periods=W)
            new[f'L{W}_{name}_mean'] = r.mean().reset_index(level=[0, 1], drop=True)
        new[f'L{W}_delaystd_max'] = g['f_delay_std_ms'].rolling(W, min_periods=W).max().reset_index(level=[0, 1], drop=True)
        new[f'L{W}_pathloss_std'] = g['f_pathloss_wmean'].rolling(W, min_periods=W).std().reset_index(level=[0, 1], drop=True)
    d = pd.concat([d, pd.DataFrame(new)], axis=1)
    return d

def load(p):
    d = add_long(pd.read_csv(p))
    d = d[(~d.meta_warmup) & d.w_complete & d.gt_label.isin(CL)].copy()
    d['y'] = d.gt_label.map({c: i for i, c in enumerate(CL)})
    return d.reset_index(drop=True)

def weights(y):
    c = np.bincount(y, minlength=3).astype(float); return (len(y) / (3 * np.maximum(c, 1)))[y]
def fit(Xtr, ytr, Xva, yva):
    p = dict(objective='multiclass', num_class=3, learning_rate=0.05, num_leaves=31, min_child_samples=50,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=0, verbose=-1, num_threads=os.cpu_count())
    dtr = lgb.Dataset(Xtr, ytr, weight=weights(ytr)); dva = lgb.Dataset(Xva, yva, weight=weights(yva), reference=dtr)
    return lgb.train(p, dtr, 800, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])

def report(name, y, yp, d):
    f1 = f1_score(y, yp, labels=[0, 1, 2], average=None, zero_division=0)
    deg = y == 1; nor = y == 0
    r = dict(macro_f1=float(f1.mean()), f1=[float(x) for x in f1], confusion=confusion_matrix(y, yp, labels=[0, 1, 2]).tolist(),
             false_congestion_on_degradation=float((yp[deg] == 2).mean()), normal_called_degradation=float((yp[nor] == 1).mean()),
             degradation_recall=float((yp[deg] == 1).mean()))
    for key in ['gt_noise_loss_pct', 'gt_noise_jitter_ms']:
        for v, idx in d.groupby(key).indices.items():
            if v > 0: r[f'recall_deg_{key}={v}'] = float((yp[idx] == 1)[y[idx] == 1].mean())
    return r

def main():
    tr = load(f'{D}/intervals_train_features.csv.gz'); va = load(f'{D}/intervals_val_features.csv.gz')
    sets = {n: load(f'{D}/intervals_{n}_features.csv.gz') for n in ['test', 'test_geant', 'test_stress']}
    base = [c for c in tr.columns if c.startswith(('f_', 'p_', 'w_', 'nb_', 'static_')) and c != 'w_complete']
    longc = [c for c in tr.columns if c.startswith('L') and c[1:3].isdigit()]
    print('base', len(base), 'long', len(longc), '| train rows', len(tr), flush=True)
    res = {}
    for name, cols in [('FULL (2 s windows)', base), ('FULL + LONG (5 s, 10 s)', base + longc)]:
        m = fit(tr[cols], tr.y, va[cols], va.y); res[name] = {}
        for sn, ds in sets.items():
            yp = m.predict(ds[cols]).argmax(1); r = report(name, ds.y.values, yp, ds); res[name][sn] = r
        r = res[name]['test']
        print(f"\n{name}: test macroF1 {r['macro_f1']:.3f} | F1 N/D/C {r['f1'][0]:.3f}/{r['f1'][1]:.3f}/{r['f1'][2]:.3f}"
              f" | degr->cong {r['false_congestion_on_degradation']:.5f} | normal->degr {r['normal_called_degradation']:.3f} | degr recall {r['degradation_recall']:.3f}")
        for k, v in r.items():
            if k.startswith('recall_deg_'): print(f'   {k} = {v:.3f}')
        for sn in ['test_geant', 'test_stress']:
            rr = res[name][sn]; print(f"   {sn}: macroF1 {rr['macro_f1']:.3f}  degr recall {rr['degradation_recall']:.3f}  normal->degr {rr['normal_called_degradation']:.3f}")
        if 'LONG' in name:
            imp = pd.Series(m.feature_importance('gain'), index=cols).sort_values(ascending=False)
            res['importance_long_top10'] = (imp / imp.sum()).head(10).round(4).to_dict(); m.save_model(f'{OUT}/lgbm_long.txt')
    json.dump(res, open(f'{OUT}/results_long.json', 'w'), indent=1); print('saved results/results_long.json')
main()
