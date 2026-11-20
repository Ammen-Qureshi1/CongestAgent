import os, sys, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
D = os.path.expanduser('~/ca_dataset/dataset'); OUT = 'results'; os.makedirs(OUT, exist_ok=True)
CL = ['normal', 'degradation', 'congestion']; CMAP = {c: i for i, c in enumerate(CL)}
SMOKE = len(sys.argv) > 2 and sys.argv[1] == '--smoke'
KEY = ['meta_run_id', 'meta_dir_id']

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

def load(p):
    d = add_long(pd.read_csv(p))
    d = d[(~d.meta_warmup) & d.w_complete].copy()          # keep 'uncertain' rows so time series stay continuous
    d['y'] = d.gt_label.map(CMAP).fillna(-1).astype(int)   # -1 = uncertain (never scored)
    return d.reset_index(drop=True)

def weights(y):
    c = np.bincount(y, minlength=3).astype(float); return (len(y) / (3 * np.maximum(c, 1)))[y]
def fit(tr, va, cols):
    a, b = tr[tr.y >= 0], va[va.y >= 0]
    p = dict(objective='multiclass', num_class=3, learning_rate=0.05, num_leaves=31, min_child_samples=50,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=0, verbose=-1, num_threads=os.cpu_count())
    dtr = lgb.Dataset(a[cols], a.y, weight=weights(a.y.values)); dva = lgb.Dataset(b[cols], b.y, weight=weights(b.y.values), reference=dtr)
    return lgb.train(p, dtr, 800, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])

def mf1(y, yp):
    m = y >= 0; f = f1_score(y[m], yp[m], labels=[0, 1, 2], average=None, zero_division=0)
    return float(f.mean()), [round(float(x), 4) for x in f]
def S(v): return '-' if v is None else f'{v:.3f}'
def rate(a):  # mean of a boolean array, None if empty
    return float(a.mean()) if len(a) else None

if SMOKE:
    d = load(sys.argv[2]); fake = (d.meta_k % 80 < 30) & (d.y == 0)   # smoke only: fake noise episodes to exercise code
    d.loc[fake, ['y', 'gt_noise_frac', 'gt_noise_loss_pct']] = [1, 1.0, 1.0]; tr = d; va = d; sets = {'test': d}
else:
    tr = load(f'{D}/intervals_train_features.csv.gz'); va = load(f'{D}/intervals_val_features.csv.gz')
    sets = {n: load(f'{D}/intervals_{n}_features.csv.gz') for n in ['test', 'test_geant', 'test_stress']}
base = [c for c in tr.columns if c.startswith(('f_', 'p_', 'w_', 'nb_', 'static_')) and c != 'w_complete']
longc = [c for c in tr.columns if c.startswith('L') and c[1:3].isdigit()]
COLS = base + longc
model = lgb.Booster(model_file=f'{OUT}/lgbm_long.txt') if os.path.exists(f'{OUT}/lgbm_long.txt') and not SMOKE else fit(tr, va, COLS)
assert list(model.feature_name()) == COLS, 'feature order mismatch with saved model'
res = {}
P = {n: model.predict(ds[COLS]) for n, ds in {**sets, 'val': va}.items()}

# ---------- CHECK 1: rule thresholds TUNED on training data (fair baseline) ----------
print('\n=== CHECK 1: best threshold rules, tuned on TRAIN (max F1 for congestion), scored on test ===', flush=True)
exc = lambda d: (d.f_delay_mean_ms - d.static_prop_delay_ms).values
GD = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40]; GL = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05]; GU = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
fam = {
 'delay only':      [(lambda d, a=a: exc(d) > a, f'delay>+{a}ms') for a in GD],
 'loss only':       [(lambda d, a=a: d.f_pathloss_wmean.values > a, f'loss>{a:.1%}') for a in GL],
 'util only':       [(lambda d, a=a: d.f_util.values > a, f'util>{a:.0%}') for a in GU],
 'delay OR loss':   [(lambda d, a=a, b=b: (exc(d) > a) | (d.f_pathloss_wmean.values > b), f'delay>+{a}ms OR loss>{b:.1%}') for a in GD for b in GL],
 'delay AND util':  [(lambda d, a=a, b=b: (exc(d) > a) & (d.f_util.values > b), f'delay>+{a}ms AND util>{b:.0%}') for a in GD for b in GU],
}
trm = tr[tr.y >= 0]; ytr = (trm.y.values == 2)
res['tuned_rules'] = {}
for fname, cands in fam.items():
    best = max(cands, key=lambda c: f1_score(ytr, c[0](trm), zero_division=0))
    out = {'rule': best[1]}
    for sn, ds in sets.items():
        m = ds[ds.y >= 0]; y = m.y.values; f = best[0](m)
        out[sn] = dict(cong_recall=rate(f[y == 2]), false_on_degr=rate(f[y == 1]), false_on_normal=rate(f[y == 0]),
                       precision=rate(y[f] == 2))
    res['tuned_rules'][fname] = out
out = {'rule': 'LightGBM (2 s + 10 s windows)'}
for sn, ds in sets.items():
    m = (ds.y >= 0).values; y = ds.y.values[m]; f = P[sn][m].argmax(1) == 2
    out[sn] = dict(cong_recall=rate(f[y == 2]), false_on_degr=rate(f[y == 1]), false_on_normal=rate(f[y == 0]), precision=rate(y[f] == 2))
res['tuned_rules']['MODEL'] = out
fmt = lambda v: '   -  ' if v is None else f'{v:6.4f}'
for sn in sets:
    print(f'-- {sn}: {"detector":46s} recall  false/degr false/norm precision')
    for k, v in res['tuned_rules'].items():
        r = v[sn]; print(f'   {v["rule"]:52s} {fmt(r["cong_recall"])} {fmt(r["false_on_degr"])}   {fmt(r["false_on_normal"])}  {fmt(r["precision"])}')

# ---------- CHECK 2: smoothing + tuned degradation threshold (no retraining) ----------
# congestion stays INSTANT (raw model); only the normal-vs-degradation choice is smoothed (causal, past only)
def decide(ds, prob, k, bias):
    pr = pd.DataFrame(prob[:, :2], columns=['pn', 'pd']); pr['g'] = ds.meta_run_id.values + '|' + ds.meta_dir_id.astype(str).values
    sm = pr.groupby('g', sort=False)[['pn', 'pd']].rolling(k, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    out = np.where(sm.pd.values * bias > sm.pn.values, 1, 0)
    out[prob.argmax(1) == 2] = 2
    return out
print('\n=== CHECK 2: smoothing over last k intervals + degradation bias, chosen on VAL ===', flush=True)
best = None
for k in [1, 2, 4, 8, 12, 20]:
    for bias in [0.5, 0.7, 0.85, 1.0, 1.2, 1.5]:
        s = mf1(va.y.values, decide(va, P['val'], k, bias))[0]
        if best is None or s > best[0]: best = (s, k, bias)
_, K, B = best
print(f'chosen on val: k={K} intervals ({K*0.25:.2f} s), bias={B}', flush=True)
res['smoothing'] = {'k': K, 'bias': B}
DEC = {}
for sn, ds in sets.items():
    raw = P[sn].argmax(1); new = decide(ds, P[sn], K, B); DEC[sn] = new; y = ds.y.values
    r = {}
    for nm, yp in [('before', raw), ('after', new)]:
        mf, f = mf1(y, yp)
        r[nm] = dict(macro_f1=mf, f1=f, normal_to_degr=rate(yp[y == 0] == 1), degr_recall=rate(yp[y == 1] == 1),
                     degr_to_cong=rate(yp[y == 1] == 2), cong_recall=rate(yp[y == 2] == 2))
        for lp in (0.5, 1.0, 2.0):
            mm = (y == 1) & (ds.gt_noise_loss_pct.values == lp)
            if mm.any(): r[nm][f'recall_loss{lp}'] = rate(yp[mm] == 1)
    res['smoothing'][sn] = r
    b, a = r['before'], r['after']
    print(f"{sn}: macroF1 {S(b['macro_f1'])} -> {S(a['macro_f1'])} | normal->degr {S(b['normal_to_degr'])} -> {S(a['normal_to_degr'])}"
          f" | degr recall {S(b['degr_recall'])} -> {S(a['degr_recall'])} | degr->cong {b['degr_to_cong']} -> {a['degr_to_cong']} | cong recall {S(a['cong_recall'])}")
    print('   loss recall before/after: ' + '  '.join(f"{k[6:]}: {b[k]:.2f}/{a[k]:.2f}" for k in b if k.startswith('recall_loss')))

# ---------- CHECK 3: episode-level view (is each noise episode caught, and how fast?) ----------
print('\n=== CHECK 3: per-EPISODE detection (an operator cares about episodes, not single 250 ms rows) ===', flush=True)
res['episodes'] = {}
for sn, ds in sets.items():
    yp = DEC[sn]; rows = []
    for (rid, did), g in ds.groupby(KEY, sort=False):
        on = (g.gt_noise_frac.values > 0); idx = np.arange(len(g)); pp = yp[g.index.values]
        st = np.flatnonzero(on & ~np.r_[False, on[:-1]]); en = np.flatnonzero(on & ~np.r_[on[1:], False])
        for s, e in zip(st, en):
            seg = pp[s:e + 1]; hit = np.flatnonzero(seg == 1)
            rows.append(dict(kind=('loss %g%%' % g.gt_noise_loss_pct.values[s:e+1].max()) if g.gt_noise_loss_pct.values[s:e+1].max() > 0
                             else ('jitter %g ms' % g.gt_noise_jitter_ms.values[s:e+1].max()),
                             dur_s=(e - s + 1) * 0.25, caught=len(hit) > 0, delay_s=hit[0] * 0.25 if len(hit) else None,
                             frac_flagged=float((seg == 1).mean())))
        # false alarm episodes on clean normal stretches
    ep = pd.DataFrame(rows)
    if len(ep) == 0: print(sn, 'no noise episodes'); continue
    ep = ep[ep.dur_s >= 5]   # ignore tiny fragments cut by warm-up / run end
    t = ep.groupby('kind').agg(episodes=('caught', 'size'), caught=('caught', 'mean'),
                               median_delay_s=('delay_s', 'median'), share_of_episode_flagged=('frac_flagged', 'mean')).round(3)
    clean = (ds.y.values == 0); on = (DEC[sn] == 1) & clean
    g = pd.Series(on).groupby((ds.meta_run_id + '|' + ds.meta_dir_id.astype(str)).values, sort=False)
    starts = int(sum((s.values & ~np.r_[False, s.values[:-1]]).sum() for _, s in g))
    fa = starts / max(clean.sum() * 0.25 / 60, 1e-9)
    res['episodes'][sn] = dict(table=t.reset_index().to_dict('records'), false_alarm_starts_per_clean_link_minute=fa)
    print(f'-- {sn}'); print(t.to_string()); print(f'   false degradation alarms that START on clean links: {fa:.2f} per link-minute')

# ---------- CHECK 4: label sensitivity (is the result an artefact of the 50% / 3-interval rule?) ----------
print('\n=== CHECK 4: change the congestion label rule; does the model still agree? ===', flush=True)
res['label_sensitivity'] = {}
def relabel(ds, T, n):
    hi = pd.Series(ds.gt_backlog_mean_frac.values > T)
    run = hi.groupby((ds.meta_run_id + '|' + ds.meta_dir_id.astype(str)).values, sort=False).rolling(n, min_periods=n).min()
    return (run.reset_index(level=0, drop=True).sort_index().fillna(0).values > 0) & (ds.gt_link_down_frac.values == 0)
for sn, ds in sets.items():
    f = P[sn].argmax(1) == 2; r = {}
    for T in (0.3, 0.5, 0.7):
        for n in (1, 3):
            c = relabel(ds, T, n); clean_noncong = (ds.gt_backlog_mean_frac.values < 0.2)
            r[f'backlog>{T:.0%} for {n}'] = dict(n_cong=int(c.sum()), recall=rate(f[c]), false_on_low_queue=rate(f[clean_noncong]))
    r['snapshot label'] = dict(n_cong=int((ds.gt_label_snapshot == 'congestion').sum()),
                               recall=rate(f[(ds.gt_label_snapshot == 'congestion').values]),
                               false_on_low_queue=rate(f[(ds.gt_label_snapshot.isin(['normal', 'degradation'])).values]))
    res['label_sensitivity'][sn] = r
    print(f'-- {sn}'); [print(f'   {k:22s} rows {v["n_cong"]:6d}  model recall {fmt(v["recall"])}  model false alarm on low queue {fmt(v["false_on_low_queue"])}') for k, v in r.items()]

# ---------- CHECK 5: no-delay ablation (how much does congestion detection lean on delay?) ----------
print('\n=== CHECK 5: retrain WITHOUT any delay/jitter feature ===', flush=True)
nod = [c for c in COLS if not any(s in c for s in ('delay', 'jitter', 'queue_delay'))]
m2 = fit(tr, va, nod); res['no_delay'] = {'n_features': len(nod)}
for sn, ds in sets.items():
    yp = m2.predict(ds[nod]).argmax(1); y = ds.y.values; mf, f = mf1(y, yp)
    res['no_delay'][sn] = dict(macro_f1=mf, f1=f, degr_to_cong=rate(yp[y == 1] == 2), cong_recall=rate(yp[y == 2] == 2))
    print(f'{sn}: macroF1 {mf:.3f} | F1 N/D/C {f} | degr->cong {res["no_delay"][sn]["degr_to_cong"]} | cong recall {res["no_delay"][sn]["cong_recall"]}')

json.dump(res, open(f'{OUT}/robustness_checks.json', 'w'), indent=1, default=str); print('\nsaved results/robustness_checks.json')
