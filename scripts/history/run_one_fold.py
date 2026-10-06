import pandas as pd, numpy as np, json, os, sys, shutil, subprocess, argparse
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
ap=argparse.ArgumentParser(); ap.add_argument('fold',type=int); a=ap.parse_args(); fold=a.fold
B=Path('/mnt/data/joel_ua'); M=B/'safe_multiscale'; O=B/'safe_stack_oof'; O.mkdir(exist_ok=True)
df=pd.read_csv(B/'pyuria_longitudinal_full.csv',low_memory=False); sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv'); trp=set(pd.to_numeric(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'],errors='coerce').dropna().astype(int)); mrn=pd.to_numeric(df.Anon_MRN,errors='coerce'); tm=mrn.isin(trp).to_numpy(); y=df.pyuria_gt5.astype(int).to_numpy()[tm]; groups=mrn[tm].astype(int).to_numpy()
meta=json.load(open(B/'safe_rich_lgbm_meta.json')); features=meta['features']; cat=meta['categorical']; X=df.loc[tm,features].reset_index(drop=True).copy()
for c in features:
    if c in cat: X[c]=X[c].fillna('__MISSING__').astype('category')
    else: X[c]=pd.to_numeric(X[c],errors='coerce')
C=pd.read_csv(M/'depth_00.csv',dtype=str,keep_default_na=False)
R=Path('/mnt/data/joel_mc/runtime'); sys.path.insert(0,str(R)); import predict_distribution as pdist
LSM=Path('/mnt/data/joel_lsm_run/LSM_static_dev-static'); os.chmod(LSM,0o755)
def ids(model): return sorted(int(p.stem.split('_')[1]) for p in (model/'trees'/'binary').glob('tree_*.bin'))
def persistence(data,model):
 r=pdist.persistence_batch(str(model/'trees'/'binary'),data.to_numpy(str),run_dir=str(model),tree_ids=ids(model),threads=8,prob_floor=1e-12); p=np.asarray(r['persistence'],float); n=np.asarray(r['n_used'],float); return np.divide(p,n,out=np.full_like(p,np.nan),where=n>0)
def target_prob(data,model):
 A=data.to_numpy(str); A[:,0]='1'; r=pdist.pseudo_code_lengths_batch(str(model/'trees'/'binary'),A,run_dir=str(model),tree_ids=ids(model),threads=8,prob_floor=1e-12,return_per_col=True); return np.power(2.,-np.asarray(r['per_col_bits'],float)[:,0])
def train_hierarchy(data,root):
 cur=data.reset_index(drop=True); models=[]
 for d in range(4):
  csv=root/f'depth_{d:02d}.csv'; model=root/f'model_{d:02d}'
  if not model.exists():
   cur.to_csv(csv,index=False); subprocess.run([str(LSM),str(csv),'0.1',str(model),'--threads','8','--subset-mode','exact'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
  models.append(model); pp=persistence(cur,model); med=np.nanmedian(pp); sc=1.4826*np.nanmedian(np.abs(pp-med)); sel=np.flatnonzero(np.isfinite(pp)&(((med-pp)/sc)>0.5)) if sc>0 else np.array([],int)
  if d<3: cur=cur.iloc[sel].reset_index(drop=True)
 return models
splits=list(GroupKFold(n_splits=5).split(np.zeros(len(y)),y,groups)); it,iv=splits[fold]
mdl=lgb.LGBMClassifier(n_estimators=500,learning_rate=.03,num_leaves=31,min_child_samples=30,subsample=.85,colsample_bytree=.8,reg_lambda=1.0,reg_alpha=.1,random_state=20261006+fold,n_jobs=8,verbosity=-1); mdl.fit(X.iloc[it],y[it],categorical_feature=cat); pl=mdl.predict_proba(X.iloc[iv])[:,1]
root=O/f'fold_{fold}'; root.mkdir(exist_ok=True); models=train_hierarchy(C.iloc[it],root); P=np.column_stack([target_prob(C.iloc[iv],models[d]) for d in range(4)])
np.savez(O/f'fold_{fold}_preds.npz',iv=iv,lgbm=pl,lsm=P,y=y[iv]); print('fold',fold,'lgbm',roc_auc_score(y[iv],pl),'lsm',[roc_auc_score(y[iv],P[:,d]) for d in range(4)],flush=True)
