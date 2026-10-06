import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
sys.path.insert(0, '/mnt/data/zedstat_pkg')
from zedstat import zedstat, calibration

OUT=Path('/mnt/data/joel_ua/zedstat_final')
OUT.mkdir(exist_ok=True)
SCORES=Path('/mnt/data/joel_ua/safe_stack_oof/lgbm_multiscale_test_scores.csv')
df=pd.read_csv(SCORES)
y=df['y'].astype(int).to_numpy(); s=df['score'].astype(float).to_numpy()
n=len(y); npos=int(y.sum()); nneg=n-npos; prev=float(y.mean())

# Raw empirical ROC from fixed held-out predictions
fpr,tpr,thr=roc_curve(y,s,drop_intermediate=False)
roc=pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':thr})
roc.to_csv(OUT/'empirical_roc.csv',index=False)

zt=zedstat.processRoc(df=roc, order=3,total_samples=n,positive_samples=npos,alpha=0.05,prevalence=prev)
zt.set_auc_bootstrap_data(df.rename(columns={'score':'score','y':'y'}),score_col='score',label_col='y',lower_score_is_risk=False)
raw_auc=float(roc_auc_score(y,s))
# bootstrap raw AUC via zedstat
raw_auc_boot=zt.auc(bootstrap=True,n_boot=2000,random_state=1729,stratified=True)

# zedstat smooth/upper hull
zt.smooth(STEP=0.001,interpolate=True,convexify=True)
hull_auc_tuple=zt.auc(total_samples=n,positive_samples=npos,alpha=0.05,bootstrap=False)
zt.allmeasures(interpolate=True)
zt.usample(precision=3,recompute_measures=True)
zt.getBounds(total_samples=n,positive_samples=npos,alpha=0.05,prevalence=prev,geometry='current')
nom=zt.get().copy()
lo=zt.df_lim['L'].copy(); hi=zt.df_lim['U'].copy()
# save full operating table joined with CIs
full=nom.copy()
for col in ['tpr','ppv','acc','npv','LR+','LR-']:
    if col in lo.columns:
        full[col+'_ci_low']=lo[col]
        full[col+'_ci_high']=hi[col]
# specificity + CI from fpr geometry (Wilson on specificity directly for selected points later)
full['specificity']=1.0-full.index.to_numpy(float)
full.to_csv(OUT/'zedstat_smoothed_hull_operating_metrics.csv')

# Youden optimum on smoothed hull
work=nom.reset_index()
work['specificity']=1-work['fpr']
work['youden_j']=work['tpr']-work['fpr']
# exclude exact endpoints
cand=work[(work.fpr>0)&(work.fpr<1)].copy()
idx=cand['youden_j'].idxmax(); op=cand.loc[idx]
fpr_op=float(op.fpr)
# nearest index in nominal bounds
ix=int(np.argmin(np.abs(nom.index.to_numpy(float)-fpr_op)))
fpr_key=float(nom.index[ix])
row=nom.iloc[ix]; lrow=lo.iloc[ix]; hrow=hi.iloc[ix]
# sensitivity CI zedstat bounds. specificity CI Wilson directly via zedstat calibration helper
sp=1-fpr_key
sp_lo,sp_hi=calibration.wilson_interval(sp*nneg,nneg,alpha=0.05)
# Note threshold interpolated on hull may be inf at endpoint only; here interior
op_summary={
 'n':n,'n_positive':npos,'n_negative':nneg,'prevalence':prev,
 'raw_auc':raw_auc,
 'raw_auc_bootstrap_nominal':float(raw_auc_boot[0]),
 'raw_auc_bootstrap_ci_low':float(raw_auc_boot[2]),
 'raw_auc_bootstrap_ci_high':float(raw_auc_boot[1]),
 'smoothed_hull_auc':float(hull_auc_tuple[0]),
 'smoothed_hull_auc_ci_low':float(hull_auc_tuple[2]),
 'smoothed_hull_auc_ci_high':float(hull_auc_tuple[1]),
 'youden_fpr':fpr_key,
 'youden_threshold':float(row['threshold']) if 'threshold' in row.index and np.isfinite(row['threshold']) else None,
 'sensitivity':float(row['tpr']),'sensitivity_ci_low':float(lrow['tpr']),'sensitivity_ci_high':float(hrow['tpr']),
 'specificity':float(sp),'specificity_ci_low':float(sp_lo),'specificity_ci_high':float(sp_hi),
 'ppv':float(row['ppv']),'ppv_ci_low':float(lrow['ppv']),'ppv_ci_high':float(hrow['ppv']),
 'npv':float(row['npv']),'npv_ci_low':float(lrow['npv']),'npv_ci_high':float(hrow['npv']),
 'accuracy':float(row['acc']),'accuracy_ci_low':float(lrow['acc']),'accuracy_ci_high':float(hrow['acc']),
 'lr_plus':float(row['LR+']) if np.isfinite(row['LR+']) else None,
 'lr_plus_ci_low':float(lrow['LR+']) if np.isfinite(lrow['LR+']) else None,
 'lr_plus_ci_high':float(hrow['LR+']) if np.isfinite(hrow['LR+']) else None,
 'lr_minus':float(row['LR-']) if np.isfinite(row['LR-']) else None,
 'lr_minus_ci_low':float(lrow['LR-']) if np.isfinite(lrow['LR-']) else None,
 'lr_minus_ci_high':float(hrow['LR-']) if np.isfinite(hrow['LR-']) else None,
 'youden_j':float(row['tpr']-fpr_key),
}

# Calibration of the fixed held-out probabilities -- no recalibration/refit.
cal_tbl=calibration.calibration_table(prob=s,y=y,n_bins=10,alpha=0.05)
cal_tbl.to_csv(OUT/'calibration_curve_10bin.csv',index=False)
calibration.plot_reliability_diagram(cal_tbl,title='Pyuria final stack: held-out calibration',outfile=OUT/'calibration_curve.png',show=False)
# raw probability calibration metrics
brier=float(brier_score_loss(y,s))
cal_intercept,cal_slope=calibration.calibration_intercept_slope(y,s)
# Bootstrap CIs using zedstat calibration bootstrap; pass raw score as both raw and probability so all metrics apply to fixed predictions
boot, ci=calibration.bootstrap_test_metrics(y_true=y,raw_score=s,calibrated_prob=s,n_boot=2000,random_state=1729)
boot.to_csv(OUT/'calibration_bootstrap_5000.csv',index=False)
op_summary.update({
 'brier_score':brier,
 'brier_score_ci_low':float(ci['brier_raw_ci'][0]),
 'brier_score_ci_high':float(ci['brier_raw_ci'][1]),
 'calibration_intercept':float(cal_intercept),
 'calibration_intercept_ci_low':float(ci['calibration_intercept_ci'][0]),
 'calibration_intercept_ci_high':float(ci['calibration_intercept_ci'][1]),
 'calibration_slope':float(cal_slope),
 'calibration_slope_ci_low':float(ci['calibration_slope_ci'][0]),
 'calibration_slope_ci_high':float(ci['calibration_slope_ci'][1]),
})

# Also extract high-specificity operating points from zedstat smoothed curve (90/95/99% specificity nearest)
ops=[]
for target_sp in [0.90,0.95,0.99]:
    target_fpr=1-target_sp
    j=int(np.argmin(np.abs(nom.index.to_numpy(float)-target_fpr)))
    fk=float(nom.index[j]); r=nom.iloc[j]; lr=lo.iloc[j]; hr=hi.iloc[j]
    sp=1-fk; spl,sph=calibration.wilson_interval(sp*nneg,nneg,alpha=.05)
    ops.append({
      'target_specificity':target_sp,'fpr':fk,'threshold':float(r['threshold']) if np.isfinite(r['threshold']) else np.nan,
      'sensitivity':float(r['tpr']),'sens_ci_low':float(lr['tpr']),'sens_ci_high':float(hr['tpr']),
      'specificity':sp,'spec_ci_low':spl,'spec_ci_high':sph,
      'ppv':float(r['ppv']),'ppv_ci_low':float(lr['ppv']),'ppv_ci_high':float(hr['ppv']),
      'npv':float(r['npv']),'npv_ci_low':float(lr['npv']),'npv_ci_high':float(hr['npv']),
      'accuracy':float(r['acc']),'acc_ci_low':float(lr['acc']),'acc_ci_high':float(hr['acc']),
      'LR+':float(r['LR+']) if np.isfinite(r['LR+']) else np.nan,'LR-':float(r['LR-']) if np.isfinite(r['LR-']) else np.nan,
    })
pd.DataFrame(ops).to_csv(OUT/'selected_high_specificity_operating_points.csv',index=False)

with open(OUT/'zedstat_summary.json','w') as f: json.dump(op_summary,f,indent=2)
print(json.dumps(op_summary,indent=2))
print('\nCalibration bins:')
print(cal_tbl.to_string(index=False))
print('\nHigh specificity points:')
print(pd.DataFrame(ops).to_string(index=False))
