"""Train + evaluate the perception classifier (LightGBM) and all ablations.
Run:  python3 train_models.py            (uses ../dataset/*_features.csv.gz)
      python3 train_models.py --smoke FILE   (quick test on one small csv)
Output: results/ folder (results.json, report.txt, predictions parquet-free csv)."""
import sys, os, json, glob, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, confusion_matrix

D = os.path.expanduser('~/ca_dataset/dataset')
OUT = 'results'; os.makedirs(OUT, exist_ok=True)
CLASSES = ['normal', 'degradation', 'congestion']
CMAP = {c: i for i, c in enumerate(CLASSES)}
SEED = 0

def load(path):
    d = pd.read_csv(path)
    d = d[(~d.meta_warmup) & d.w_complete & d.gt_label.isin(CLASSES)].copy()
    d['y'] = d.gt_label.map(CMAP)
    return d

# ---- feature groups (gt_* and meta_* are NEVER inputs) ----
def feats(d):
    return [c for c in d.columns if c.startswith(('f_', 'p_', 'w_', 'nb_', 'static_'))
            and c not in ('w_complete',)]
def sel(cols, drop_sub=(), keep_only=None):
    if keep_only is not None: return [c for c in cols if c in keep_only]
    return [c for c in cols if not any(s in c for s in drop_sub)]

def groups(cols):
    LOSS = ('loss', 'probe_lost', 'dropped')
    TREND = ('trend', 'delay_d1', 'delay_d2', 'corr_')
    return {
      'FULL': cols,
      'A2_no_trend_corr': sel(cols, TREND),
      'A3_no_loss': sel(cols, LOSS),
      'A4_no_neighbour': sel(cols, ('nb_',)),
      'A5_no_queue_est_util_corr': sel(cols, ('queue_delay_est', 'corr_', 'util')),
      'B3like_delay_loss_only': sel(cols, keep_only=[c for c in cols if c in (
          'p_delay_ms','p_delay_ema','p_loss_ema','f_delay_mean_ms','f_pathloss_wmean','f_probe_loss_ratio')]),
      'util_only': sel(cols, keep_only=['f_util', 'w_util_mean', 'w_util_max']),
    }

def weights(y):
    cnt = np.bincount(y, minlength=3).astype(float)
    w = len(y) / (3 * np.maximum(cnt, 1))
    return w[y]

def fit(Xtr, ytr, Xva, yva, binary=False):
    p = dict(objective='binary' if binary else 'multiclass', learning_rate=0.05, num_leaves=31,
             min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             seed=SEED, verbose=-1, num_threads=os.cpu_count())
    if not binary: p['num_class'] = 3
    dtr = lgb.Dataset(Xtr, ytr, weight=weights(ytr))
    dva = lgb.Dataset(Xva, yva, weight=weights(yva), reference=dtr)
    return lgb.train(p, dtr, 600, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(30, verbose=False)])

def predict(m, X, binary=False):
    p = m.predict(X)
    return (p > 0.5).astype(int) if binary else p.argmax(1)

def metrics(y, yp, d):
    cm = confusion_matrix(y, yp, labels=[0, 1, 2])
    f1 = f1_score(y, yp, labels=[0, 1, 2], average=None, zero_division=0)
    deg = (y == 1); nor = (y == 0); con = (y == 2)
    r = dict(macro_f1=float(f1.mean()), f1_normal=float(f1[0]), f1_degradation=float(f1[1]),
             f1_congestion=float(f1[2]), confusion=cm.tolist(),
             # RQ1 key numbers: how often a NON-congestion state is called congestion
             false_congestion_on_degradation=float((yp[deg] == 2).mean()) if deg.any() else None,
             false_congestion_on_normal=float((yp[nor] == 2).mean()) if nor.any() else None,
             congestion_recall=float((yp[con] == 2).mean()) if con.any() else None,
             congestion_precision=float((y[yp == 2] == 2).mean()) if (yp == 2).any() else None,
             n=int(len(y)))
    runs = []
    for rid, g in d.groupby('meta_run_id'):
        idx = g.index.values
        runs.append(f1_score(y[np.isin(d.index.values, idx)], yp[np.isin(d.index.values, idx)],
                             labels=[0, 1, 2], average='macro', zero_division=0))
    r['per_run_macro_f1_mean'] = float(np.mean(runs)); r['per_run_macro_f1_std'] = float(np.std(runs))
    r['n_runs'] = len(runs)
    return r

def main():
    smoke = len(sys.argv) > 2 and sys.argv[1] == '--smoke'
    if smoke:
        d = load(sys.argv[2]); tr, va = d.iloc[:1500], d.iloc[1500:2200]
        sets = {'test': d.iloc[2200:]}
    else:
        tr = load(f'{D}/intervals_train_features.csv.gz'); va = load(f'{D}/intervals_val_features.csv.gz')
        sets = {n: load(f'{D}/intervals_{n}_features.csv.gz') for n in ['test', 'test_geant', 'test_stress']}
    cols = feats(tr); G = groups(cols)
    print('features:', len(cols), '| train rows', len(tr), '| classes', np.bincount(tr.y, minlength=3).tolist(), flush=True)
    res = {'n_features': len(cols), 'feature_groups': {k: len(v) for k, v in G.items()}, 'models': {}}
    lines = []
    preds_out = {}
    for name, fc in G.items():
        t = time.time()
        m = fit(tr[fc], tr.y, va[fc], va.y)
        res['models'][name] = {'best_iter': m.best_iteration, 'splits': {}}
        for sn, ds in sets.items():
            yp = predict(m, ds[fc]); r = metrics(ds.y.values, yp, ds.reset_index(drop=True).set_index(ds.index))
            res['models'][name]['splits'][sn] = r
            if name == 'FULL': preds_out[sn] = yp
        r = res['models'][name]['splits']['test']
        print(f"{name:28s} test macroF1 {r['macro_f1']:.3f} | f1 N/D/C {r['f1_normal']:.3f}/{r['f1_degradation']:.3f}/{r['f1_congestion']:.3f}"
              f" | degr->cong {r['false_congestion_on_degradation']} | {time.time()-t:.0f}s", flush=True)
        if name == 'FULL':
            imp = pd.Series(m.feature_importance('gain'), index=fc).sort_values(ascending=False)
            res['feature_importance_gain_top20'] = (imp / imp.sum()).head(20).round(4).to_dict()
            m.save_model(f'{OUT}/lgbm_full.txt')
    # A1: no non-congestive class (congestion vs everything else) -> degradation looks like congestion?
    ytr1 = (tr.y == 2).astype(int); yva1 = (va.y == 2).astype(int)
    m1 = fit(tr[cols], ytr1, va[cols], yva1, binary=True)
    res['models']['A1_binary_no_noncongestive_class'] = {'splits': {}}
    for sn, ds in sets.items():
        p = (m1.predict(ds[cols]) > 0.5).astype(int) * 2   # 2 = congestion, 0 = not
        r = metrics(ds.y.values, p, ds)
        res['models']['A1_binary_no_noncongestive_class']['splits'][sn] = r
    r = res['models']['A1_binary_no_noncongestive_class']['splits']['test']
    print(f"{'A1_binary (no degr. class)':28s} test degr->cong {r['false_congestion_on_degradation']} | cong recall {r['congestion_recall']}", flush=True)
    # breakdowns for FULL on test: by noise type, load
    te = sets['test'].copy(); te['pred'] = preds_out['test']
    br = {}
    for key in ['meta_noise', 'meta_load']:
        for v, g in te.groupby(key):
            br[f'{key}={v}'] = dict(n=int(len(g)), macro_f1=float(f1_score(g.y, g.pred, labels=[0,1,2], average='macro', zero_division=0)),
                                   degr_called_cong=float(((g.y == 1) & (g.pred == 2)).sum() / max((g.y == 1).sum(), 1)))
    res['full_breakdown_test'] = br
    json.dump(res, open(f'{OUT}/results.json', 'w'), indent=1)
    print('saved', f'{OUT}/results.json')

if __name__ == '__main__':
    main()
