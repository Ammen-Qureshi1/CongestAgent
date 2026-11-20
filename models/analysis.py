"""RQ1 offline evidence: simple threshold rules vs the trained model.
Counts how often each detector says CONGESTION on rows that are NOT congestion
(= would trigger an unnecessary reroute). Run: python3 analysis.py"""
import os, sys, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
D = os.path.expanduser('~/ca_dataset/dataset')
CL = ['normal', 'degradation', 'congestion']
def load(p):
    d = pd.read_csv(p)
    d = d[(~d.meta_warmup) & d.w_complete & d.gt_label.isin(CL)].copy()
    d['y'] = d.gt_label.map({c: i for i, c in enumerate(CL)}); return d.reset_index(drop=True)
def feats(d):
    return [c for c in d.columns if c.startswith(('f_', 'p_', 'w_', 'nb_', 'static_')) and c != 'w_complete']
smoke = len(sys.argv) > 2 and sys.argv[1] == '--smoke'
sets = {'test': load(sys.argv[2])} if smoke else {n: load(f'{D}/intervals_{n}_features.csv.gz') for n in ['test', 'test_geant', 'test_stress']}
model = None if smoke else lgb.Booster(model_file='results/lgbm_full.txt')
excess = lambda d: d.f_delay_mean_ms - d.static_prop_delay_ms
rules = {
 'R1 delay > +3 ms': lambda d: excess(d) > 3,
 'R2 delay > +10 ms': lambda d: excess(d) > 10,
 'R3 loss > 0.5%': lambda d: d.f_pathloss_wmean > 0.005,
 'R4 loss > 2%': lambda d: d.f_pathloss_wmean > 0.02,
 'R5 util > 80%': lambda d: d.f_util > 0.8,
 'R6 (delay>+3ms) OR (loss>0.5%)': lambda d: (excess(d) > 3) | (d.f_pathloss_wmean > 0.005),
 'R7 (delay>+10ms) OR (loss>2%)': lambda d: (excess(d) > 10) | (d.f_pathloss_wmean > 0.02),
}
res = {}
for sn, d in sets.items():
    y = d.y.values; out = {}
    dets = {k: f(d).values for k, f in rules.items()}
    if model is not None:
        X = d[model.feature_name()]; dets['MODEL (LightGBM full)'] = model.predict(X).argmax(1) == 2
    for k, flag in dets.items():
        r = dict(flag_rate_on_congestion=float(flag[y == 2].mean()),
                 false_flag_on_degradation=float(flag[y == 1].mean()),
                 false_flag_on_normal=float(flag[y == 0].mean()),
                 false_flag_on_all_noncongestion=float(flag[y != 2].mean()),
                 precision=float((y[flag] == 2).mean()) if flag.any() else None)
        out[k] = r
    res[sn] = out
    print(f'\n== {sn}  (rows {len(d)}; congestion {int((y==2).sum())}, degradation {int((y==1).sum())}, normal {int((y==0).sum())})')
    print(f"{'detector':34s} {'cong.recall':>11s} {'false/degr':>10s} {'false/norm':>10s} {'precision':>9s}")
    for k, r in out.items():
        p = '-' if r['precision'] is None else f"{r['precision']:.3f}"
        print(f"{k:34s} {r['flag_rate_on_congestion']:11.3f} {r['false_flag_on_degradation']:10.4f} {r['false_flag_on_normal']:10.4f} {p:>9s}")
# degradation-vs-normal detail for the full model (test split)
if model is not None:
    d = sets['test']; pr = model.predict(d[model.feature_name()]).argmax(1)
    det = {}
    for key in ['meta_noise', 'gt_noise_loss_pct', 'gt_noise_jitter_ms', 'meta_load']:
        for v, g in d.assign(pred=pr).groupby(key):
            idx = g.index
            det[f'{key}={v}'] = dict(n=int(len(g)),
                recall_degradation=float(((g.pred == 1) & (g.y == 1)).sum() / max((g.y == 1).sum(), 1)) if (g.y == 1).any() else None,
                normal_called_degradation=float(((g.pred == 1) & (g.y == 0)).sum() / max((g.y == 0).sum(), 1)) if (g.y == 0).any() else None)
    res['detail_full_model_test'] = det
    print('\nFULL model detail (test): degradation recall / normal wrongly called degradation')
    for k, v in det.items(): print(f'{k:28s} n={v["n"]:6d} deg_recall={v["recall_degradation"]} normal->deg={v["normal_called_degradation"]}')
json.dump(res, open('results/analysis.json', 'w'), indent=1)
