import pandas as pd, numpy as np, sys, os, json
from pathlib import Path
from sklearn.metrics import roc_auc_score
B=Path('/mnt/data/joel_ua/safe_multiscale'); R=Path('/mnt/data/joel_mc/runtime'); sys.path.insert(0,str(R)); import predict_distribution as p

def score(df,model):
 X=df.to_numpy(dtype=str); X[:,0]='1'
 ids=sorted(int(x.stem.split('_')[1]) for x in (model/'trees'/'binary').glob('tree_*.bin'))
 r=p.pseudo_code_lengths_batch(str(model/'trees'/'binary'),X,run_dir=str(model),tree_ids=ids,threads=8,prob_floor=1e-12,return_per_col=True)
 bits=np.asarray(r['per_col_bits'],float)[:,0]
 return np.power(2.0,-bits)
tr=pd.read_csv(B/'depth_00.csv',dtype=str,keep_default_na=False); te=pd.read_csv(B/'test.csv',dtype=str,keep_default_na=False)
ytr=tr.iloc[:,0].astype(int).to_numpy(); yte=te.iloc[:,0].astype(int).to_numpy()
outtr={}; outte={}
for d in range(4):
 m=B/f'model_depth_{d:02d}'; pt=score(tr,m); pe=score(te,m); outtr[f'lsm_p{d}']=pt; outte[f'lsm_p{d}']=pe
 print('depth',d,'train_auc',roc_auc_score(ytr,pt),'test_auc',roc_auc_score(yte,pe), 'mean',pe.mean(),flush=True)
pd.DataFrame(outtr).to_csv(B/'train_multiscale_probs.csv',index=False); pd.DataFrame(outte).to_csv(B/'test_multiscale_probs.csv',index=False)
