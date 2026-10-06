import pandas as pd, numpy as np, json, os, sys, shutil, subprocess
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
B=Path('/mnt/data/joel_ua'); M=B/'safe_multiscale'; O=B/'safe_stack_oof'; O.mkdir(exist_ok=True)
# full rich data and masks
df=pd.read_csv(B/'pyuria_longitudinal_full.csv',low_memory=False)
sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv'); trp=set(pd.to_numeric(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'],errors='coerce').dropna().astype(int))
mrn=pd.to_numeric(df.Anon_MRN,errors='coerce'); train_mask=mrn.isin(trp).to_numpy(); test_mask=~train_mask
y=df.pyuria_gt5.astype(int).to_numpy(); ytr=y[train_mask]; yte=y[test_mask]; groups=mrn[train_mask].astype(int).to_numpy()
# LightGBM feature config
meta=json.load(open(B/'safe_rich_lgbm_meta.json')); features=meta['features']; cat=meta['categorical']; X=df[features].copy()
for c in features:
    if c in cat: X[c]=X[c].fillna('__MISSING__').astype('category')
    else: X[c]=pd.to_numeric(X[c],errors='coerce')
Xtr=X.loc[train_mask].reset_index(drop=True); Xte=X.loc[test_mask].reset_index(drop=True)
# categorical LSM data
Ctr=pd.read_csv(M/'depth_00.csv',dtype=str,keep_default_na=False); Cte=pd.read_csv(M/'test.csv',dtype=str,keep_default_na=False)
# runtime
R=Path('/mnt/data/joel_mc/runtime'); sys.path.insert(0,str(R)); import predict_distribution as pdist
LSM=Path('/mnt/data/joel_lsm_run/LSM_static_dev-static'); os.chmod(LSM,0o755)

def ids(model): return sorted(int(p.stem.split('_')[1]) for p in (model/'trees'/'binary').glob('tree_*.bin'))
def persistence(data,model):
 r=pdist.persistence_batch(str(model/'trees'/'binary'),data.to_numpy(str),run_dir=str(model),tree_ids=ids(model),threads=8,prob_floor=1e-12); p=np.asarray(r['persistence'],float); n=np.asarray(r['n_used'],float); return np.divide(p,n,out=np.full_like(p,np.nan),where=n>0)
def target_prob(data,model):
 A=data.to_numpy(str); A[:,0]='1'; r=pdist.pseudo_code_lengths_batch(str(model/'trees'/'binary'),A,run_dir=str(model),tree_ids=ids(model),threads=8,prob_floor=1e-12,return_per_col=True); bits=np.asarray(r['per_col_bits'],float)[:,0]; return np.power(2.,-bits)
def train_hierarchy(data,root):
 cur=data.reset_index(drop=True); models=[]
 for d in range(4):
  csv=root/f'depth_{d:02d}.csv'; model=root/f'model_{d:02d}'; cur.to_csv(csv,index=False)
  if model.exists(): shutil.rmtree(model)
  subprocess.run([str(LSM),str(csv),'0.1',str(model),'--threads','8','--subset-mode','exact'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
  models.append(model); pp=persistence(cur,model); med=np.nanmedian(pp); sc=1.4826*np.nanmedian(np.abs(pp-med)); sel=np.flatnonzero(np.isfinite(pp)&(((med-pp)/sc)>0.5)) if sc>0 else np.array([],int)
  if d<3:
   if len(sel)<100: break
   cur=cur.iloc[sel].reset_index(drop=True)
 return models

gkf=GroupKFold(n_splits=5); oof_lgbm=np.zeros(len(ytr)); oof_lsm=np.zeros((len(ytr),4))
for fold,(it,iv) in enumerate(gkf.split(np.zeros(len(ytr)),ytr,groups)):
 print('FOLD',fold,'train',len(it),'val',len(iv),flush=True)
 mdl=lgb.LGBMClassifier(n_estimators=500,learning_rate=.03,num_leaves=31,min_child_samples=30,subsample=.85,colsample_bytree=.8,reg_lambda=1.0,reg_alpha=.1,random_state=20261006+fold,n_jobs=8,verbosity=-1)
 mdl.fit(Xtr.iloc[it],ytr[it],categorical_feature=cat); oof_lgbm[iv]=mdl.predict_proba(Xtr.iloc[iv])[:,1]
 root=O/f'fold_{fold}'; root.mkdir(exist_ok=True); models=train_hierarchy(Ctr.iloc[it],root)
 for d in range(4):
  model=models[min(d,len(models)-1)]
  oof_lsm[iv,d]=target_prob(Ctr.iloc[iv],model)
 print(' fold auc lgbm',roc_auc_score(ytr[iv],oof_lgbm[iv]),'lsm', [roc_auc_score(ytr[iv],oof_lsm[iv,d]) for d in range(4)],flush=True)
np.save(O/'oof_lgbm.npy',oof_lgbm); np.save(O/'oof_lsm.npy',oof_lsm)
# Test full-model scores
ptest_lgbm=pd.read_csv(B/'safe_rich_lgbm_test_scores.csv').lgbm_prob.to_numpy(float)
ptest_lsm=pd.read_csv(M/'test_multiscale_probs.csv').to_numpy(float)
# transformations
clip=lambda a:np.clip(a,1e-5,1-1e-5)
logit=lambda a:np.log(clip(a)/(1-clip(a)))
variants={'lgbm':['lgbm'],'lgbm_p0':['lgbm','p0'],'lgbm_multiscale':['lgbm','p0','p1','p2','p3']}
train_cols={'lgbm':logit(oof_lgbm),'p0':logit(oof_lsm[:,0]),'p1':logit(oof_lsm[:,1]),'p2':logit(oof_lsm[:,2]),'p3':logit(oof_lsm[:,3])}
test_cols={'lgbm':logit(ptest_lgbm),'p0':logit(ptest_lsm[:,0]),'p1':logit(ptest_lsm[:,1]),'p2':logit(ptest_lsm[:,2]),'p3':logit(ptest_lsm[:,3])}
rows=[]
for name,cols in variants.items():
 A=np.column_stack([train_cols[c] for c in cols]); T=np.column_stack([test_cols[c] for c in cols]); sc=StandardScaler().fit(A); Az=sc.transform(A); Tz=sc.transform(T)
 lr=LogisticRegressionCV(Cs=np.logspace(-2,2,9),cv=5,scoring='roc_auc',max_iter=2000,random_state=20261006).fit(Az,ytr)
 po=lr.predict_proba(Az)[:,1]; pt=lr.predict_proba(Tz)[:,1]
 rows.append({'variant':name,'oof_auc':roc_auc_score(ytr,po),'test_auc':roc_auc_score(yte,pt),'C':float(lr.C_[0]),'coef':lr.coef_[0].tolist(),'features':cols})
 print(name,'OOF',rows[-1]['oof_auc'],'TEST',rows[-1]['test_auc'],'C',rows[-1]['C'],'coef',rows[-1]['coef'],flush=True)
 pd.DataFrame({'row_index':np.flatnonzero(test_mask),'y':yte,'score':pt}).to_csv(O/f'{name}_test_scores.csv',index=False)
json.dump(rows,open(O/'stack_results.json','w'),indent=2)
print('BASE test lgbm',roc_auc_score(yte,ptest_lgbm),flush=True)
