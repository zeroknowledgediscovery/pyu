from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'
FIG=RES/'figures'
FIG.mkdir(parents=True, exist_ok=True)

# ROC: empirical + zedstat smoothed upper hull
emp=pd.read_csv(RES/'zedstat'/'empirical_roc.csv')
hull=pd.read_csv(RES/'zedstat'/'zedstat_smoothed_hull_operating_metrics.csv')
summary=json.load(open(RES/'zedstat'/'zedstat_summary.json'))
fig,ax=plt.subplots(figsize=(6.4,5.6))
ax.plot(emp.fpr, emp.tpr, linewidth=1.2, alpha=.75, label=f"Empirical ROC (AUC={summary['raw_auc']:.3f})")
ax.plot(hull.fpr, hull.tpr, linewidth=2.0, label=f"zedstat smoothed hull (AUC={summary['smoothed_hull_auc']:.3f})")
ax.plot([0,1],[0,1],'--',linewidth=1,label='Chance')
ax.fill_between(hull.fpr, hull.tpr_ci_low, hull.tpr_ci_high, alpha=.12, linewidth=0)
ax.set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0,1), ylim=(0,1), title='Pyuria prediction: ROC performance')
ax.legend(frameon=False, loc='lower right')
ax.grid(alpha=.2)
fig.tight_layout()
for ext in ('png','pdf'): fig.savefig(FIG/f'roc_curve_zedstat.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)

# Precision-recall curve derived from zedstat operating table at observed prevalence.
pr=hull.replace([np.inf,-np.inf],np.nan).dropna(subset=['tpr','ppv']).copy()
pr=pr.sort_values('tpr')
fig,ax=plt.subplots(figsize=(6.4,5.6))
ax.plot(pr.tpr, pr.ppv, linewidth=2.0, label='zedstat-derived PR curve')
if {'ppv_ci_low','ppv_ci_high'}.issubset(pr.columns):
    ax.fill_between(pr.tpr, pr.ppv_ci_low, pr.ppv_ci_high, alpha=.12, linewidth=0)
ax.axhline(summary['prevalence'], linestyle='--', linewidth=1, label=f"Prevalence={summary['prevalence']:.3f}")
ax.set(xlabel='Recall (sensitivity)', ylabel='Precision (PPV)', xlim=(0,1), ylim=(0,1), title='Pyuria prediction: precision-recall')
ax.legend(frameon=False, loc='upper right')
ax.grid(alpha=.2)
fig.tight_layout()
for ext in ('png','pdf'): fig.savefig(FIG/f'precision_recall_zedstat.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)

# Calibration from zedstat bins
cal=pd.read_csv(RES/'zedstat'/'calibration_curve_10bin.csv')
fig,ax=plt.subplots(figsize=(5.8,5.6))
ax.plot([0,0.8],[0,0.8],'--',linewidth=1,label='Ideal')
yerr=np.vstack([cal.obs_rate-cal.obs_rate_lo, cal.obs_rate_hi-cal.obs_rate])
ax.errorbar(cal.mean_pred,cal.obs_rate,yerr=yerr,fmt='o-',capsize=3,linewidth=1.5,label='Observed')
ax.set(xlabel='Predicted probability',ylabel='Observed event rate',xlim=(0,0.8),ylim=(0,0.8),title='Calibration curve')
ax.legend(frameon=False)
ax.grid(alpha=.2)
fig.tight_layout()
for ext in ('png','pdf'): fig.savefig(FIG/f'calibration_curve_zedstat.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)

# Feature importance: gain
gain=pd.read_csv(RES/'tables'/'lightgbm_feature_importance_gain.csv').head(20).sort_values('gain')
fig,ax=plt.subplots(figsize=(7.2,7.2))
ax.barh(gain.feature,gain.gain_fraction)
ax.set(xlabel='Fraction of total LightGBM gain',title='Top LightGBM features by gain')
ax.grid(axis='x',alpha=.2)
fig.tight_layout()
for ext in ('png','pdf'): fig.savefig(FIG/f'feature_importance_gain.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)

# Feature importance: held-out SHAP
shap=pd.read_csv(RES/'tables'/'lightgbm_feature_importance_shap.csv').head(20).sort_values('mean_abs_shap')
fig,ax=plt.subplots(figsize=(7.2,7.2))
ax.barh(shap.feature,shap.mean_abs_shap)
ax.set(xlabel='Mean |SHAP contribution| on held-out test set',title='Top features by held-out SHAP magnitude')
ax.grid(axis='x',alpha=.2)
fig.tight_layout()
for ext in ('png','pdf'): fig.savefig(FIG/f'feature_importance_shap.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)

# Model comparison
perf=json.load(open(RES/'final_performance.json'))
rows=[]
for r in perf['stack_results']:
    rows.append((r['variant'],r['test_auc']))
order=['lsm_multiscale','lgbm','lgbm_p0','lgbm_p01','lgbm_multiscale']
d=dict(rows)
labels=['LSM multiscale','LightGBM','LightGBM + M0','LightGBM + M0/M1','LightGBM + M0-M3']
vals=[d[x] for x in order]
fig,ax=plt.subplots(figsize=(7.0,4.7))
ax.bar(labels,vals)
ax.set_ylim(0.68,0.74)
ax.set_ylabel('Held-out AUC')
ax.set_title('Incremental value of multiscale LSM stacking')
ax.tick_params(axis='x',rotation=25)
for i,v in enumerate(vals): ax.text(i,v+0.0007,f'{v:.3f}',ha='center',fontsize=9)
ax.grid(axis='y',alpha=.2)
fig.tight_layout()
for ext in ('png','pdf'): fig.savefig(FIG/f'model_comparison.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)

print('Wrote figures to',FIG)
