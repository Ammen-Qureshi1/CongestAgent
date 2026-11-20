import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#555','axes.linewidth':0.6,'figure.dpi':300})
d = pd.read_csv('../models/results/figures/figure_data.csv')
COL = {'none':'#9a9a9a','delay':'#2a78d6','delayloss':'#eb6834','delayutil':'#1baf7a','model':'#eda100','model_robust':'#e87ba4',
       'central':'#6b6b6b','regional':'#2a78d6','regional_nofast':'#7fb2ea','regional_v2':'#0b4f9e'}
NAME = {'none':'No control','delay':'Delay rule','delayloss':'Delay+loss rule','delayutil':'Delay+util rule','model':'CA (gated)','model_robust':'CA (robust)',
        'central':'Central','regional':'Regional','regional_nofast':'Regional\n(no fast loop)','regional_v2':'Regional v2'}
def get(fig, metric):
    x = d[(d.figure==fig)&(d.metric==metric)]; return x.set_index('controller')
def bars(ax, fig, metric, order, title, ylabel=None, fmt='{:.3f}', ylim=None):
    x = get(fig, metric).loc[order]
    xs = np.arange(len(order))
    ax.bar(xs, x['mean'], yerr=x['ci95'], color=[COL[o] for o in order], width=0.62, capsize=2, error_kw=dict(lw=0.7, ecolor='#333'))
    ax.set_xticks(xs); ax.set_xticklabels([NAME[o] for o in order], rotation=30, ha='right', fontsize=(7 if 'central' in order else 8))
    ax.set_title(title, fontsize=8.5, loc='left'); 
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim) if ylim else ax.set_ylim(0, float((x['mean'] + x['ci95']).max()) * 1.15)
    lo, hi = ax.get_ylim()
    for i,(m) in enumerate(x['mean']): ax.text(i, m + x['ci95'].iloc[i] + 0.025*(hi-lo), fmt.format(m), ha='center', fontsize=6.5)
    ax.grid(axis='y', lw=0.3, color='#ddd'); ax.set_axisbelow(True)
five = ['delay','delayloss','delayutil','model','model_robust']

# Fig 2: RQ1 false reroutes (two panels)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={'width_ratios':[1,1.35]})
bars(ax[0], 'fig1', 'false_reroute_rate', five, '(a) False-reroute rate (30 runs each)', 'Share of reroutes on degraded links', ylim=(0, 0.19))
x = d[(d.figure=='fig2')].copy(); x['noise'] = x.metric.str.extract(r'\[(.*)\]')
noises = ['loss2','jitter40','shift40','shift80']; w = 0.16
for i,c in enumerate(five):
    s = x[x.controller==c].set_index('noise').loc[noises]
    ax[1].bar(np.arange(4)+(i-2)*w, s['mean'], w, yerr=s['ci95'], color=COL[c], label=NAME[c], capsize=1.2, error_kw=dict(lw=0.5, ecolor='#333'))
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(['random loss 2%','jitter 40 ms','delay shift +40 ms','delay shift +80 ms'], rotation=15, ha='right')
ax[1].set_title('(b) By impairment type (6 runs each)', fontsize=8.5, loc='left'); ax[1].grid(axis='y', lw=0.3, color='#ddd'); ax[1].set_axisbelow(True)
ax[1].legend(fontsize=6, frameon=False, ncol=2, loc='upper left')
plt.tight_layout(); plt.savefig('figures/fig_rq1.png'); plt.close()

# Fig 3: detection trade-off
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
bars(ax[0], 'fig3', 'flag_rate_degradation', five, '(a) Degraded intervals flagged as congestion', 'Share of degraded intervals', '{:.3f}')
bars(ax[1], 'fig3', 'flag_recall_congestion', five, '(b) Congestion intervals flagged (recall)', 'Recall', '{:.3f}', ylim=(0.9, 1.035))
plt.tight_layout(); plt.savefig('figures/fig_detect.png'); plt.close()

# Fig 4: performance
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6))
six = ['none']+five
bars(ax[0], 'fig4', 'path_delay_mean_ms', six, '(a) Mean path delay (ms)', None, '{:.1f}', ylim=(0, 90))
bars(ax[1], 'fig4', 'tcp_throughput_mbps', six, '(b) TCP throughput (Mbps)', None, '{:.1f}')
bars(ax[2], 'fig4', 'fct_p95_s', six, '(c) p95 flow completion (s)', None, '{:.2f}')
plt.tight_layout(); plt.savefig('figures/fig_perf.png'); plt.close()

NAME.update({'regional_nofast':'No fast loop','regional_v2':'Regional v2'})
# Fig 5: RQ2
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.7))
r = ['central','regional','regional_nofast','regional_v2']
bars(ax[0],'fig5','longhaul_msgs_per_s',r,'(a) Cross-region msgs/s',None,'{:.1f}')
bars(ax[1],'fig5','latency_ms_mean',r,'(b) Mean latency (ms)',None,'{:.1f}')
bars(ax[2],'fig5','latency_ms_p95',r,'(c) p95 latency (ms)',None,'{:.0f}')
bars(ax[3],'fig5','reroutes',r,'(d) Reroutes per run',None,'{:.0f}')
plt.tight_layout(); plt.savefig('figures/fig_rq2.png'); plt.close()

# Fig 6: fairness
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
bars(ax[0],'fig6','jain_cubic_vs_bbr',six,"(a) Jain's index, CUBIC vs BBR",None,'{:.3f}',ylim=(0,1.0))
bars(ax[1],'fig6','bbr_over_cubic',six,'(b) BBR / CUBIC throughput ratio',None,'{:.2f}',ylim=(0,5))
plt.tight_layout(); plt.savefig('figures/fig_rq3.png'); plt.close()

# Fig 1: architecture
fig, ax = plt.subplots(figsize=(7.0, 3.3)); ax.axis('off'); ax.set_xlim(-1.5,101.5); ax.set_ylim(-6,44)
def box(x,y,w,h,t,fc='#eef3fa',ec='#2a78d6',fs=6.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.2',fc=fc,ec=ec,lw=0.9)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(x1,y1,x2,y2,t=None,dx=0,dy=1.6,line=False):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-' if line else '-|>',mutation_scale=8,lw=0.9,color='#333'))
    if t: ax.text((x1+x2)/2+dx,(y1+y2)/2+dy,t,ha='center',fontsize=6,color='#333')
box(0,16,22,12,'Emulated network\n(Mininet, OVS, tc)\nTCP CUBIC/BBR, UDP,\nimpairments',fc='#f3f3f3',ec='#777')
box(27,16,23,12,'Telemetry, every 250 ms\nutilisation, delay, loss,\n5 s and 10 s windows')
box(52,31,30,11,'Perception (LightGBM)\nnormal / congestion / degradation\n+ busy-link gate',fs=6.2)
box(55,2,24,12,'Regional agents\nfast loop 250 ms\nslow loop 2 s\n(summaries, revert)',fc='#fdf3df',ec='#eda100')
box(82,16,18,12,'Reroute action\nprimary / backup\npath (ovs-ofctl)',fc='#e9f6f0',ec='#1baf7a')
arr(22.5,22,26.5,22)
arr(50.5,26,55,33,'features',dx=-6.5,dy=1.5)
arr(67,30.5,67,14.6,'link states',dx=6,dy=0)
arr(79.5,8,88,15.6,'decisions',dx=2.3,dy=-4.8)
arr(91,15.6,91,-3,None,line=True); arr(91,-3,11,-3,None,line=True); arr(11,-3,11,15.6,'new paths',dx=6,dy=-4.2)
ax.text(50,-5.5,'Closed loop. The simulator has the same structure, with PPO agents and an emulated perception stage.',ha='center',fontsize=6.3,style='italic',color='#444')
plt.savefig('figures/fig_arch.png', bbox_inches='tight'); plt.close()
print('ok')
