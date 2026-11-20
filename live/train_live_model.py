"""
Retrain the perception model on the FULL dataset (incl. harder noise) for the live runs, and re-tune the
rule baselines on the same training data so the comparison stays fair.
  python3 train_live_model.py   -> ../models/results/lgbm_live.txt, live_rules.json, live_model_report.json
"""
import os, json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import f1_score
HERE = Path(__file__).resolve().parent
D = HERE.parent / 'dataset'; OUT = HERE.parent / 'models' / 'results'; OUT.mkdir(parents=True, exist_ok=True)
CL = ['normal', 'degradation', 'congestion']; KEY = ['meta_run_id', 'meta_dir_id']


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


def load(name):
    d = add_long(pd.read_csv(D / f'intervals_{name}_features.csv.gz'))
    d = d[(~d.meta_warmup) & d.w_complete & d.gt_label.isin(CL)].copy()
    d['y'] = d.gt_label.map({c: i for i, c in enumerate(CL)})
    return d.reset_index(drop=True)


def weights(y):
    c = np.bincount(y, minlength=3).astype(float); return (len(y) / (3 * np.maximum(c, 1)))[y]


tr, va = load('train'), load('val')
te = {n: load(n) for n in ('test', 'test_geant', 'test_stress')}
cols = [c for c in tr.columns if c.startswith(('f_', 'p_', 'w_', 'nb_', 'static_')) and c != 'w_complete'] + \
       [c for c in tr.columns if c.startswith('L') and c[1:3].isdigit()]
print('train runs', tr.meta_run_id.nunique(), '| rows', len(tr), '| noise types', sorted(tr.meta_noise.unique()), flush=True)
p = dict(objective='multiclass', num_class=3, learning_rate=0.05, num_leaves=31, min_child_samples=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, seed=0, verbose=-1, num_threads=os.cpu_count())
dtr = lgb.Dataset(tr[cols], tr.y, weight=weights(tr.y.values)); dva = lgb.Dataset(va[cols], va.y, weight=weights(va.y.values), reference=dtr)
m = lgb.train(p, dtr, 800, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
m.save_model(str(OUT / 'lgbm_live.txt'))
# re-tune the rules on TRAIN (max F1 of congestion vs rest)
ex = lambda d: (d.f_delay_mean_ms - d.static_prop_delay_ms).values
yt = tr.y.values == 2
GD = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]; GL = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
bd = max(GD, key=lambda a: f1_score(yt, ex(tr) > a))
bdl = max(((a, b) for a in GD for b in GL), key=lambda ab: f1_score(yt, (ex(tr) > ab[0]) | (tr.f_pathloss_wmean.values > ab[1])))
# utilisation gate: a queue can only build on a busy link. Level chosen on TRAIN only: keeps 99% of congestion rows.
UG = float(tr[tr.y == 2].w_util_mean.quantile(0.01))
json.dump({'delay_ms': bd, 'loss': bdl[1], 'delayloss_delay_ms': bdl[0], 'util_gate': UG,
           'note': 'tuned on train split (max congestion F1); util_gate = 1% quantile of w_util_mean on train congestion rows'},
          open(OUT / 'live_rules.json', 'w'), indent=1)
print(f'rules re-tuned on train: delay > +{bd} ms | delay > +{bdl[0]} ms OR loss > {bdl[1]:.0%} | util gate w_util_mean >= {UG:.3f}')
rep = {}
print(f"\n{'split / noise':28s} {'rows':>7s} {'macroF1':>7s} | false CONGESTION on degradation:  model(gated)  delay-rule  delay|loss-rule  delay+util-rule | cong recall: model  delay+util")
for sn, d in te.items():
    P = m.predict(d[cols]); pr = P.argmax(1); y = d.y.values
    gate = d.w_util_mean.values >= UG
    bad = (pr == 2) & ~gate; pr[bad] = P[bad][:, :2].argmax(1)          # gated model: no 'congestion' on a quiet link
    rd = ex(d) > bd; rdl = (ex(d) > bdl[0]) | (d.f_pathloss_wmean.values > bdl[1]); rdu = (ex(d) > bd) & gate
    for nz, idx in [('ALL', np.arange(len(d)))] + sorted(d.groupby('meta_noise').indices.items()):
        yy, pp = y[idx], pr[idx]; deg = yy == 1
        r = dict(rows=int(len(idx)), macro_f1=float(f1_score(yy, pp, labels=[0, 1, 2], average='macro', zero_division=0)),
                 model_false_cong=float((pp[deg] == 2).mean()) if deg.any() else None,
                 delay_rule_false_cong=float(rd[idx][deg].mean()) if deg.any() else None,
                 delayloss_rule_false_cong=float(rdl[idx][deg].mean()) if deg.any() else None,
                 delayutil_rule_false_cong=float(rdu[idx][deg].mean()) if deg.any() else None,
                 delayutil_rule_cong_recall=float(rdu[idx][yy == 2].mean()) if (yy == 2).any() else None,
                 model_cong_recall=float((pp[yy == 2] == 2).mean()) if (yy == 2).any() else None,
                 delay_rule_cong_recall=float(rd[idx][yy == 2].mean()) if (yy == 2).any() else None)
        rep[f'{sn}/{nz}'] = r
        f = lambda v: '   -  ' if v is None else f'{v:6.3f}'
        print(f"{sn + ' / ' + nz:28s} {r['rows']:7d} {r['macro_f1']:7.3f} |                                  {f(r['model_false_cong'])}        {f(r['delay_rule_false_cong'])}     {f(r['delayloss_rule_false_cong'])}        {f(r['delayutil_rule_false_cong'])}      | {f(r['model_cong_recall'])}  {f(r['delayutil_rule_cong_recall'])}")
json.dump(rep, open(OUT / 'live_model_report.json', 'w'), indent=1)
print('\nsaved lgbm_live.txt, live_rules.json, live_model_report.json')
