"""
Offline check of the robust model (CongestAgent/train_perception_colab.py), with and without the busy-link gate, on the NSFNET test split.
  python3 check_robust.py          (about 5-10 min; do not run during a live batch)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_perception_colab as tp          # same feature code as used to train the robust model
D = HERE.parent / 'dataset'; R = HERE.parent / 'models' / 'results'
G = json.loads((R / 'live_rules.json').read_text())['util_gate']
m = lgb.Booster(model_file=str(R / 'robust' / 'model.txt')); F = json.loads((R / 'robust' / 'features.json').read_text())
rows = []
def h(d, base, lc):
    P = m.predict(d[F])
    x = pd.DataFrame(dict(y=d.y.values, noise=d.meta_noise.values, p=P.argmax(1), u=d.w_util_mean.values))
    g = x.p.values.copy(); bad = (g == 2) & (x.u.values < G); g[bad] = P[bad][:, :2].argmax(1); x['g'] = g
    rows.append(x)
tp.stream(str(D / 'intervals_test_features.csv.gz'), 'robust', False, 300000, h)
r = pd.concat(rows, ignore_index=True)
print(f'gate: w_util_mean >= {G:.3f}.  For comparison, our gated model: test macroF1 0.930, false-cong 0.004, recall 0.990; shift80 false-cong 0.055')
print(f"{'test split':14s} {'macroF1 (no gate -> gate)':>28s} {'false-cong on noise':>24s} {'congestion recall':>22s}")
for nz, d in [('ALL', r)] + [(n, r[r.noise == n]) for n in ('shift80', 'shift40', 'jitter40', 'loss2', 'loss0.5')]:
    deg, c = d.y == 1, d.y == 2
    f = lambda col: f1_score(d.y, d[col], labels=[0, 1, 2], average='macro', zero_division=0)
    fc = lambda col: (d[col][deg] == 2).mean() if deg.any() else float('nan')
    rc = lambda col: (d[col][c] == 2).mean() if c.any() else float('nan')
    print(f"{nz:14s} {f('p'):12.3f} -> {f('g'):.3f} {fc('p'):14.4f} -> {fc('g'):.4f} {rc('p'):12.3f} -> {rc('g'):.3f}")
