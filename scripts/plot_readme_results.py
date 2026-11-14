"""Generate README figures from archived experiment metrics. Run from any directory."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/results/chart_data'
OUT=ROOT/'docs/results/assets/charts';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})
colors=['#2563eb','#f59e0b','#14b8a6']
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
s=json.loads((DATA/'evaluation_summary.json').read_text())['validation']
v=[json.loads(x) for x in (DATA/'validation.jsonl').read_text().splitlines() if x.strip()]
r=list(csv.DictReader((DATA/'training_metrics.csv').open()))
x=np.array([int(a['step']) for a in r]);y=np.array([float(a['loss']) for a in r])
fig,axs=plt.subplots(1,2,figsize=(11,4.2));ax=axs[0]
ax.plot(x,y,color=colors[0],lw=.7,alpha=.35,label='Logged loss (100-step averages)')
window=20
ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,'valid'),color=colors[0],lw=2,label='20-point moving average')
ax.set_yscale('log');ax.set(xlabel='Training step',ylabel='Training objective (log scale)',title='Training objective');ax.legend(fontsize=8)
ax=axs[1]
for key,c,label in [('body_mae',colors[0],'First five joints'),('gripper_mae',colors[1],'Gripper')]:ax.plot([a['step'] for a in v],[a[key] for a in v],'o-',color=c,label=label)
best=min(v,key=lambda a:a['body_mae']);ax.axvline(best['step'],color='#64748b',ls='--',lw=1,label='Lowest body MAE checkpoint')
ax.set(xlabel='Checkpoint step',ylabel='MAE (dataset joint units)',title='Checkpoint validation: full chunks');ax.legend(fontsize=8)
fig.suptitle('ACT / 180 training episodes / 100k steps',fontsize=14,fontweight='bold');fig.tight_layout(rect=(0,.04,1,.94));fig.text(.5,.015,'Training loss and validation MAE use different definitions and scales.',ha='center',fontsize=9,color='#475569');save(fig,'training_validation')
h=sorted((int(k),a) for k,a in s['horizons'].items());fig,ax=plt.subplots(figsize=(8,4.4))
for key,c,label in [('body_mae',colors[0],'First five joints'),('gripper_mae',colors[1],'Gripper')]:ax.plot([k for k,a in h],[a[key] for k,a in h],'o-',color=c,lw=2,label=label)
ax.set(xlabel='Action index within predicted chunk (0-based)',ylabel='MAE (dataset joint units)',title='Prediction error across the action chunk',ylim=(0,8));ax.legend();fig.text(.5,.01,'60k checkpoint | 20 validation episodes | targets beyond episode end excluded',ha='center',fontsize=8);fig.tight_layout(rect=(0,.04,1,1));save(fig,'prediction_horizon')
fig,ax=plt.subplots(figsize=(9,4.5));modes=['fresh_first','temporal_0.01','steps32'];loc=np.arange(3);w=.25
for i,(label,fun,c) in enumerate([('First five joints',lambda a:a['body_mae'],colors[0]),('Gripper',lambda a:a['gripper_mae'],colors[1]),('Gripper: closing window',lambda a:a['closing_window']['gripper_mae'],colors[2])]):
 vals=[fun(s['modes'][m]) for m in modes];bars=ax.bar(loc+(i-1)*w,vals,w,label=label,color=c);ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
ax.set_xticks(loc,['Latest first action','Temporal ensemble (0.01)','Execute 32 actions']);ax.set(ylabel='MAE (dataset joint units)',title='Execution strategy comparison / offline replay',ylim=(0,10.5));ax.legend(fontsize=8,ncol=3);fig.text(.5,.01,'20 validation episodes / 9,447 frames | demonstration observations, not closed-loop success',ha='center',fontsize=8);fig.tight_layout(rect=(0,.04,1,1));save(fig,'strategy_comparison')
fig,axs=plt.subplots(2,2,figsize=(12,7),sharex='col')
for col,ep in enumerate([12,99]):
 a=np.load(DATA/f'episode_{ep:06d}.npz');t=np.arange(len(a['target']))/30
 for row,(j,label) in enumerate([(1,'Shoulder lift'),(5,'Gripper')]):
  ax=axs[row,col];ax.plot(t,a['target'][:,j],color='#0f172a',lw=2,label='Demonstration target')
  for key,c,lab in [('temporal',colors[1],'Temporal ensemble'),('steps32',colors[2],'Execute 32 actions')]:ax.plot(t,a[key][:,j],color=c,lw=1,alpha=.85,label=lab)
  ax.set(title=f'Validation episode {ep} / {label}',ylabel='Target (dataset joint units)')
  if row==1:ax.set_xlabel('Time (s, dataset 30 Hz)')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False);fig.suptitle('Joint target traces under recorded observations',fontweight='bold');fig.tight_layout(rect=(0,.05,1,.95));save(fig,'joint_traces')
fig,ax=plt.subplots(figsize=(8,4.4));sweep=[(int(k),a) for k,a in s['execution_sweep'].items() if a['interior_jump_median'] is not None];sweep.sort()
for key,c,label in [('boundary_jump_median',colors[0],'Chunk boundary'),('interior_jump_median',colors[2],'Within chunk')]:ax.plot([k for k,a in sweep],[a[key] for k,a in sweep],'o-',color=c,lw=2,label=label)
ax.set(xlabel='Number of actions executed per query (K)',ylabel='Median target jump (dataset joint units)',title='Target discontinuity at chunk boundaries',ylim=(0,8));ax.legend();fig.tight_layout();save(fig,'chunk_discontinuity')
print('Generated 5 PNG + 5 SVG figures')
