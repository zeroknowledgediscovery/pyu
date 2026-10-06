import numpy as np,pandas as pd,json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
B=Path('/mnt/data/joel_ua'); O=B/'safe_stack_oof'; M=B/'safe_multiscale'
df=pd.read_csv(B/'pyuria_longitudinal_full.csv',usecols=['Anon_MRN','pyuria_gt5']); sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv'); trp=set(pd.to_numeric(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'],errors='coerce').dropna().astype(int)); mrn=pd.to_numeric(df.Anon_MRN,errors='coerce'); tm=mrn.isin(trp).to_numpy(); y=df.pyuria_gt5.astype(int).to_numpy(); ytr=y[tm]; yte=y[~tm]
o_l=np.zeros(len(ytr)); o_p=np.zeros((len(ytr),4))
for f in range(5):
 z=np.load(O/f'fold_{f}_preds.npz'); iv=z['iv']; o_l[iv]=z['lgbm']; o_p[iv]=z['lsm']
t_l=pd.read_csv(B/'safe_rich_lgbm_test_scores.csv').lgbm_prob.to_numpy(float); t_p=pd.read_csv(M/'test_multiscale_probs.csv').to_numpy(float)
clip=lambda a:np.clip(a,1e-5,1-1e-5); logit=lambda a:np.log(clip(a)/(1-clip(a)))
train={'lgbm':logit(o_l),**{f'p{i}':logit(o_p[:,i]) for i in range(4)}}; test={'lgbm':logit(t_l),**{f'p{i}':logit(t_p[:,i]) for i in range(4)}}
variants={'lgbm':['lgbm'],'lgbm_p0':['lgbm','p0'],'lgbm_p01':['lgbm','p0','p1'],'lgbm_multiscale':['lgbm','p0','p1','p2','p3'],'lsm_multiscale':['p0','p1','p2','p3']}
res=[]
for name,cols in variants.items():
 A=np.column_stack([train[c] for c in cols]); T=np.column_stack([test[c] for c in cols]); sc=StandardScaler().fit(A); Az=sc.transform(A); Tz=sc.transform(T)
 lr=LogisticRegressionCV(Cs=np.logspace(-3,3,13),cv=5,scoring='roc_auc',max_iter=5000,random_state=20261006).fit(Az,ytr)
 po=lr.predict_proba(Az)[:,1]; pt=lr.predict_proba(Tz)[:,1]
 r={'variant':name,'features':cols,'oof_auc':roc_auc_score(ytr,po),'test_auc':roc_auc_score(yte,pt),'C':float(lr.C_[0]),'coef':lr.coef_[0].tolist(),'intercept':lr.intercept_.tolist()}; res.append(r); print(r)
 pd.DataFrame({'y':yte,'score':pt}).to_csv(O/f'{name}_test_scores.csv',index=False)
print('OOF raw lgbm',roc_auc_score(ytr,o_l),'test raw',roc_auc_score(yte,t_l)); print('OOF p0',roc_auc_score(ytr,o_p[:,0]),'test p0',roc_auc_score(yte,t_p[:,0]))
json.dump(res,open(O/'stack_results.json','w'),indent=2)
