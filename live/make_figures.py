"""
Paper figures from the live results (PNG at 300 dpi + PDF). Run after live_eval.py, rq2_eval.py and live_extra.py.
  python3 make_figures.py      -> ../models/results/figures/fig*.png / .pdf  and figure_data.csv
Every bar = mean over runs, error bar = 95% confidence interval (t), the unit is one run.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'models' / 'results'
OUT = RES / 'figures'; OUT.mkdir(parents=True, exist_ok=True)

# one fixed colour per controller (validated categorical palette, assigned in slot order; 'none' = neutral grey)
C = {'none': '#8a8984', 'delay': '#2a78d6', 'delayloss': '#eb6834', 'delayutil': '#1baf7a', 'model': '#eda100',
     'model_robust': '#e87ba4', 'central': '#2a78d6', 'regional': '#eb6834', 'regional_nofast': '#1baf7a', 'regional_v2': '#eda100'}
NAME = {'none': 'No control', 'delay': 'Delay rule', 'delayloss': 'Delay+loss rule', 'delayutil': 'Delay+util rule',
        'model': 'CA (gated)', 'model_robust': 'CA (robust)',
        'central': 'Central', 'regional': 'Regional', 'regional_nofast': 'Regional, no fast loop', 'regional_v2': 'Regional v2'}
# CA = CongestAgent perception model; "gated" = our LightGBM + busy-link gate; "robust" = robust LightGBM + gate
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'figure.dpi': 110, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'})
LOG = []


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return (v.mean() if len(v) else np.nan), 0.0
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def bars(ax, df, pols, metric, ylabel, fmt='{:.3f}', title=None):
    pols = [p for p in pols if (df.policy == p).any()]
    ms, hs = zip(*[ci(df[df.policy == p][metric]) for p in pols]) if pols else ([], [])
    x = np.arange(len(pols))
    ax.bar(x, ms, width=0.62, color=[C[p] for p in pols], edgecolor='white', linewidth=2, zorder=2)
    ax.errorbar(x, ms, yerr=hs, fmt='none', ecolor='#0b0b0b', elinewidth=1, capsize=3, zorder=3)
    for xi, m, h in zip(x, ms, hs):
        ax.text(xi, m + h, fmt.format(m), ha='center', va='bottom', fontsize=6.5 if len(pols) > 5 else 7.5, color='#0b0b0b')
    ax.set_xticks(x); ax.set_xticklabels([NAME[p] for p in pols], fontsize=7, rotation=30, ha='right', rotation_mode='anchor')
    ax.set_ylabel(ylabel); ax.grid(axis='x', visible=False)
    ax.set_ylim(0, max([m + h for m, h in zip(ms, hs)] + [1e-9]) * 1.18)
    if title:
        ax.set_title(title, fontsize=9, loc='left', color='#0b0b0b')
    for p, m, h in zip(pols, ms, hs):
        LOG.append(dict(figure=ax.figure._label, metric=metric, controller=p, mean=m, ci95=h, n=int((df.policy == p).sum())))


def save(fig, name, caption):
    fig.tight_layout(); fig.savefig(OUT / f'{name}.png'); fig.savefig(OUT / f'{name}.pdf'); plt.close(fig)
    print(f'{name}: {caption}')


RQ1 = ['none', 'delay', 'delayloss', 'delayutil', 'model', 'model_robust']
L = pd.read_csv(RES / 'live_per_run.csv')
L = L[L.policy.isin(RQ1)]

# ---- Fig 1: RQ1 false reroutes
fig = plt.figure(figsize=(7.2, 2.9)); fig._label = 'fig1'
ax1, ax2 = fig.subplots(1, 2)
bars(ax1, L, RQ1[1:], 'false_reroute_rate', 'False-reroute rate', title='(a) Reroutes triggered on a noisy link')
bars(ax2, L, RQ1[1:], 'reroutes_on_degradation', 'Reroutes per run', '{:.1f}', title='(b) Reroutes on noisy links per run')
save(fig, 'fig1_rq1_false_reroutes', 'RQ1, live: false reroutes per controller (mean, 95% CI, 30 runs each)')

# ---- Fig 2: RQ1 false-reroute rate by noise type (grouped bars)
pols = [p for p in RQ1[1:] if (L.policy == p).any()]
noises = [n for n in ('loss2', 'jitter40', 'shift40', 'shift80') if (L.noise == n).any()]
fig, ax = plt.subplots(figsize=(7.2, 2.9)); fig._label = 'fig2'
w = 0.8 / max(len(pols), 1)
for i, p in enumerate(pols):
    ms, hs = zip(*[ci(L[(L.policy == p) & (L.noise == n)].false_reroute_rate) for n in noises])
    xs = np.arange(len(noises)) + (i - (len(pols) - 1) / 2) * w
    ax.bar(xs, ms, width=w, color=C[p], edgecolor='white', linewidth=1.5, label=NAME[p], zorder=2)
    ax.errorbar(xs, ms, yerr=hs, fmt='none', ecolor='#0b0b0b', elinewidth=0.8, capsize=2, zorder=3)
    for n, m, h in zip(noises, ms, hs):
        LOG.append(dict(figure='fig2', metric=f'false_reroute_rate[{n}]', controller=p, mean=m, ci95=h))
ax.set_xticks(np.arange(len(noises))); ax.set_xticklabels(['Random loss 2%', 'Jitter 40 ms', 'Delay shift +40 ms', 'Delay shift +80 ms'][:len(noises)] if noises == ['loss2', 'jitter40', 'shift40', 'shift80'] else noises)
ax.set_ylabel('False-reroute rate'); ax.grid(axis='x', visible=False)
ax.legend(frameon=False, fontsize=7, ncol=3, loc='upper left')
save(fig, 'fig2_rq1_false_reroutes_by_noise', 'RQ1, live: false-reroute rate by noise type (6 runs per bar)')

# ---- Fig 3: RQ1 detection trade-off: noise flagged as congestion vs real congestion caught
fig = plt.figure(figsize=(7.2, 2.9)); fig._label = 'fig3'
ax1, ax2 = fig.subplots(1, 2)
bars(ax1, L[L.noise != 'none'], RQ1[1:], 'flag_rate_degradation', 'Share of noisy intervals', title='(a) Noise flagged as congestion (lower is better)')
bars(ax2, L, RQ1[1:], 'flag_recall_congestion', 'Share of congested intervals', title='(b) Real congestion detected (higher is better)')
ax2.set_ylim(0.8, 1.04)
save(fig, 'fig3_rq1_detection', 'RQ1, live: detection trade-off of each controller')

# ---- Fig 4: RQ1 network performance
fig = plt.figure(figsize=(7.2, 2.9)); fig._label = 'fig4'
axs = fig.subplots(1, 3)
bars(axs[0], L, RQ1, 'path_delay_mean_ms', 'ms', '{:.1f}', title='(a) Mean path delay')
bars(axs[1], L, RQ1, 'tcp_throughput_mbps', 'Mbit/s', '{:.1f}', title='(b) TCP throughput')
bars(axs[2], L, RQ1, 'fct_p95_s', 's', '{:.2f}', title='(c) p95 flow completion time')
save(fig, 'fig4_rq1_performance', 'RQ1, live: delay, throughput and flow completion time per controller')

# ---- Fig 5: RQ2 regional vs central
R2p = RES / 'rq2_per_run.csv'
if R2p.exists():
    R2 = pd.read_csv(R2p)
    P2 = ['central', 'regional', 'regional_nofast', 'regional_v2']
    fig = plt.figure(figsize=(7.2, 5.0)); fig._label = 'fig5'
    axs = fig.subplots(2, 2).ravel()
    bars(axs[0], R2, P2, 'latency_ms_mean', 'ms', '{:.1f}', title='(a) Mean decision latency')
    bars(axs[1], R2, P2, 'latency_ms_p95', 'ms', '{:.1f}', title='(b) p95 decision latency')
    bars(axs[2], R2, P2, 'longhaul_msgs_per_s', 'messages / s', '{:.1f}', title='(c) Control messages crossing regions')
    bars(axs[3], R2, P2, 'reroutes', 'reroutes per run', '{:.0f}', title='(d) Reroutes per run')
    save(fig, 'fig5_rq2_regional_vs_central', 'RQ2, live: regional agents vs one central agent (22 matched runs each, NSFNET + GEANT)')

# ---- Fig 6: RQ3 fairness under control
R3p = RES / 'live_rq3_per_run.csv'
if R3p.exists():
    R3 = pd.read_csv(R3p); R3 = R3[R3.policy.isin(RQ1)]
    fig = plt.figure(figsize=(7.2, 2.8)); fig._label = 'fig6'
    ax1, ax2 = fig.subplots(1, 2)
    bars(ax1, R3, RQ1, 'jain_cubic_vs_bbr', "Jain's index", title="(a) CUBIC vs BBR fairness on busy links")
    ax1.set_ylim(0.5, 1.0); ax1.axhline(1.0, color='#52514e', lw=0.8, ls=':')
    bars(ax2, R3, RQ1, 'bbr_over_cubic', 'BBR / CUBIC goodput', '{:.2f}', title='(b) BBR advantage (median ratio per run)')
    ax2.axhline(1.0, color='#52514e', lw=0.8, ls=':')
    save(fig, 'fig6_rq3_fairness', 'RQ3, live: fairness between CUBIC and BBR under each controller')

# ---- Fig 7: offline perception, noise called congestion by noise type (test split)
rp = RES / 'live_model_report.json'
if rp.exists():
    rep = json.loads(rp.read_text())
    ks = [k for k in ('loss0.5', 'loss1', 'loss2', 'jitter5', 'jitter10', 'jitter20', 'jitter40', 'shift40', 'shift80') if f'test/{k}' in rep]
    series = [('delay_rule_false_cong', 'Delay rule', C['delay']), ('delayloss_rule_false_cong', 'Delay+loss rule', C['delayloss']),
              ('delayutil_rule_false_cong', 'Delay+util rule', C['delayutil']), ('model_false_cong', 'CA (gated)', C['model'])]
    fig, ax = plt.subplots(figsize=(7.2, 2.8)); fig._label = 'fig7'
    w = 0.8 / len(series)
    for i, (key, lab, col) in enumerate(series):
        v = [rep[f'test/{k}'][key] or 0 for k in ks]
        ax.bar(np.arange(len(ks)) + (i - 1.5) * w, v, width=w, color=col, edgecolor='white', linewidth=1.2, label=lab, zorder=2)
        for k, x in zip(ks, v):
            LOG.append(dict(figure='fig7', metric=f'false_cong[test/{k}]', controller=lab, mean=x))
    ax.set_xticks(np.arange(len(ks))); ax.set_xticklabels(ks, fontsize=8)
    ax.set_ylabel('Noisy intervals called congestion'); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper left')
    save(fig, 'fig7_offline_false_congestion', 'Offline (NSFNET test split): share of noisy intervals called congestion, by noise type')

pd.DataFrame(LOG).to_csv(OUT / 'figure_data.csv', index=False)
print(f'\nsaved {len(list(OUT.glob("*.png")))} figures and figure_data.csv in {OUT}')
