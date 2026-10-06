import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/mnt/data/joel_mc/runtime'); import predict_distribution as p
B=Path('/mnt/data/joel_ua'); O=B/'final_multiscale'; seed=20261006
D=pd.read_csv(B/'pyuria_longitudinal_full.csv',low_memory=False); y=D.pyuria_gt5.astype(int).to_numpy(); mrn=D.Anon_MRN.astype(int).to_numpy(); sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv'); train_set=set(pd.to_numeric(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'],errors='coerce').dropna().astype(int)); rng=np.random.default_rng(seed+1); u=np.array(sorted(train_set)); perm=rng.permutation(u); nbase=int(np.floor(.8*len(u))); base_m=set(perm[:nbase]); stack_m=set(perm[nbase:]); base_mask=np.array([m in base_m for m in mrn]); stack_mask=np.array([m in stack_m for m in mrn])
# lgb
meta=json.load(open(B/'longitudinal_lgbm_meta.json')); feats=meta['features']; X=D[feats].copy(); cat=[]
for c in feats:
 if X[c].dtype=='object': X[c]=X[c].fillna('').astype('category'); cat.append(c)
 else:X[c]=pd.to_numeric(X[c],errors='coerce')
params=dict(n_estimators=700,learning_rate=.025,num_leaves=31,subsample=.9,colsample_bytree=.8,reg_lambda=1.5,reg_alpha=.2,min_child_samples=30,random_state=seed,n_jobs=8,verbosity=-1)
model=LGBMClassifier(**params).fit(X.loc[base_mask],y[base_mask],categorical_feature=cat); pl=model.predict_proba(X.loc[stack_mask])[:,1]
# categorical LSM matrix using saved transform
trf=json.load(open(O/'lsm_transform.json')); C=pd.DataFrame(index=D.index); C['pyuria_gt5']=D.pyuria_gt5.astype(int).astype(str)
for c in trf['features']:
 spec=trf['transform'][c]; s=D[c]
 if spec['type']=='cat':
  v=s.fillna('').astype(str); C[c]=v.where(v.isin(spec['top']),'OTHER')
 elif spec['type']=='raw': C[c]=pd.to_numeric(s,errors='coerce').map(lambda v:'<NA>' if pd.isna(v) else str(v))
 else:
  e=np.array(spec['edges'],float); labs=[f'Q{i+1}' for i in range(len(e)-1)]; C[c]=pd.cut(pd.to_numeric(s,errors='coerce'),bins=e,labels=labs,include_lowest=True).astype(str).replace('nan','<NA>')
C=C.fillna('<NA>').astype(str); rows=C.loc[stack_mask]; A=rows.to_numpy(dtype=str).copy();A[:,0]=''; P=[]
for d in range(4):
 m=O/'internal'/f'model_{d:02d}'; vals=np.empty(len(A)); t=time.time()
 for i,row in enumerate(A): vals[i]=float(p.predict_distribution(str(m/'trees'/'binary'),0,row,raw=True,run_dir=str(m)).get('1',0.0))
 print('depth',d,'sec',time.time()-t,'auc',roc_auc_score(y[stack_mask],vals),flush=True);P.append(vals)
P=np.column_stack(P); clip=lambda q:np.clip(q,1e-5,1-1e-5); Z=np.column_stack([np.log(clip(pl)/(1-clip(pl)))]+[np.log(clip(P[:,j])/(1-clip(P[:,j]))) for j in range(4)]); mu=Z.mean(0);sd=Z.std(0);sd[sd==0]=1; lr=LogisticRegression(C=.5,max_iter=2000).fit((Z-mu)/sd,y[stack_mask]); ps=lr.predict_proba((Z-mu)/sd)[:,1]; print('stackval',roc_auc_score(y[stack_mask],ps),'coef',lr.coef_[0],flush=True)
json.dump({'mu':mu.tolist(),'sd':sd.tolist(),'coef':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0]),'stackval_auc':float(roc_auc_score(y[stack_mask],ps))},open(O/'stack_meta.json','w'),indent=2);np.save(O/'P_stack.npy',P)
