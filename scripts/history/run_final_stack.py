import os,sys,json,subprocess,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,'/mnt/data/joel_mc/runtime')
import predict_distribution as pdist
B=Path('/mnt/data/joel_ua'); R=Path('/mnt/data/joel_mc/runtime'); OUT=B/'final_multiscale'; OUT.mkdir(exist_ok=True)
seed=20261006
# raw data/splits
D=pd.read_csv(B/'pyuria_longitudinal_full.csv',low_memory=False); y=D.pyuria_gt5.astype(int).to_numpy(); mrn=D.Anon_MRN.astype(int).to_numpy()
sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv'); train_m=np.array(pd.to_numeric(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'],errors='coerce').dropna().astype(int)); train_set=set(train_m); tr=np.array([m in train_set for m in mrn]); te=~tr
# choose raw predictors and fit final LightGBM + internal base-fit LightGBM
meta=json.load(open(B/'longitudinal_lgbm_meta.json')); feats=meta['features']; X=D[feats].copy(); cat=[]
for c in feats:
    if X[c].dtype=='object': X[c]=X[c].fillna('').astype('category'); cat.append(c)
    else: X[c]=pd.to_numeric(X[c],errors='coerce')
params=dict(n_estimators=700,learning_rate=.025,num_leaves=31,subsample=.9,colsample_bytree=.8,reg_lambda=1.5,reg_alpha=.2,min_child_samples=30,random_state=seed,n_jobs=8,verbosity=-1)
full_lgb=LGBMClassifier(**params).fit(X.loc[tr],y[tr],categorical_feature=cat); p_lgb_test=full_lgb.predict_proba(X.loc[te])[:,1]
# internal patient split within preserved training
rng=np.random.default_rng(seed+1); u=np.array(sorted(train_set)); perm=rng.permutation(u); nbase=int(np.floor(.8*len(u))); base_m=set(perm[:nbase]); stack_m=set(perm[nbase:]); base_mask=np.array([m in base_m for m in mrn]); stack_mask=np.array([m in stack_m for m in mrn])
base_lgb=LGBMClassifier(**params).fit(X.loc[base_mask],y[base_mask],categorical_feature=cat); p_lgb_stack=base_lgb.predict_proba(X.loc[stack_mask])[:,1]
print('LGBM test',roc_auc_score(y[te],p_lgb_test),'stackval',roc_auc_score(y[stack_mask],p_lgb_stack),flush=True)
# top 50 gain features from corrected final LGBM; exclude problematic high-card ZIP3 from LSM
imp=pd.DataFrame({'feature':feats,'gain':full_lgb.booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False)
sel=[]
for c in imp.feature:
    if c in {'ZIP3'}: continue
    sel.append(c)
    if len(sel)>=50: break
print('LSM features',sel,flush=True)
# categorical transform fitted on all preserved training only
C=pd.DataFrame(index=D.index); C['pyuria_gt5']=D.pyuria_gt5.astype(int).astype(str); transform={}
for c in sel:
    s=D[c]
    if s.dtype=='object':
        vals=s.loc[tr].fillna('').astype(str); tops=vals.value_counts().head(20).index.tolist(); C[c]=s.fillna('').astype(str).where(s.fillna('').astype(str).isin(tops),'OTHER'); transform[c]={'type':'cat','top':tops}
    else:
        z=pd.to_numeric(s,errors='coerce'); ztr=z[tr]; uniq=ztr.dropna().nunique()
        if uniq<=12:
            C[c]=z.map(lambda v:'<NA>' if pd.isna(v) else str(v)); transform[c]={'type':'raw'}
        else:
            try:
                _,edges=pd.qcut(ztr,q=8,retbins=True,duplicates='drop'); edges=np.unique(edges); edges[0]=-np.inf; edges[-1]=np.inf
                lab=[f'Q{i+1}' for i in range(len(edges)-1)]; C[c]=pd.cut(z,bins=edges,labels=lab,include_lowest=True).astype(str).replace('nan','<NA>'); transform[c]={'type':'qbin','edges':[float(x) for x in edges]}
            except Exception:
                C[c]=z.map(lambda v:'<NA>' if pd.isna(v) else str(v)); transform[c]={'type':'raw'}
C=C.fillna('<NA>').astype(str)
json.dump({'features':sel,'transform':transform},open(OUT/'lsm_transform.json','w'),indent=2)
# hierarchy trainer
BIN=str(R/'LSM_static_dev-static')
def train_hierarchy(df,name):
    root=OUT/name
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    levels=[]; models=[]; current=df.reset_index(drop=True); hist=[]
    for d in range(4):
        csv=root/f'depth_{d:02d}.csv'; current.to_csv(csv,index=False); md=root/f'model_{d:02d}'
        cmd=[BIN,str(csv),'0.1',str(md),'--threads','5','--subset-mode','exact']
        t=time.time(); subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT); print(name,'trained depth',d,'n',len(current),'sec',time.time()-t,flush=True)
        levels.append(current); models.append(md)
        trees=sorted(int(p.stem.split('_')[1]) for p in (md/'trees'/'binary').glob('tree_*.bin'))
        res=pdist.persistence_batch(str(md/'trees'/'binary'),current.to_numpy(dtype=str),run_dir=str(md),tree_ids=trees,threads=5,prob_floor=1e-12)
        pp=np.asarray(res['persistence'],float); used=np.asarray(res['n_used'],float); avg=np.divide(pp,used,out=np.full(len(pp),np.nan),where=used>0); med=np.nanmedian(avg); scale=1.4826*np.nanmedian(np.abs(avg-med)); z=(med-avg)/scale if scale>0 else np.full(len(avg),np.nan); idx=np.flatnonzero(np.isfinite(z)&(z>.5)); hist.append({'depth':d,'n':len(current),'next':len(idx),'median':float(med),'scale':float(scale)})
        if d==3 or len(idx)<150: break
        current=current.iloc[idx].reset_index(drop=True)
    json.dump(hist,open(root/'hierarchy.json','w'),indent=2); return models

def target_probs(models,rows,label):
    Xr=rows.to_numpy(dtype=str).copy(); Xr[:,0]=''; out=[]
    for d,m in enumerate(models):
        vals=np.empty(len(Xr),float); t=time.time()
        for i,row in enumerate(Xr):
            dist=pdist.predict_distribution(str(m/'trees'/'binary'),0,row,raw=True,run_dir=str(m)); vals[i]=float(dist.get('1',0.0))
        print(label,'depth',d,'pred sec',time.time()-t,'auc',roc_auc_score(y[rows.index],vals),flush=True); out.append(vals)
    return np.column_stack(out)
# internal hierarchy on base-fit patients
mods_internal=train_hierarchy(C.loc[base_mask],'internal')
P_stack=target_probs(mods_internal,C.loc[stack_mask], 'stack')
# fit stack on internal heldout patients only. logit transform probabilities + L2 regularization
clip=lambda p: np.clip(p,1e-5,1-1e-5)
Zs=np.column_stack([np.log(clip(p_lgb_stack)/(1-clip(p_lgb_stack))), *[np.log(clip(P_stack[:,j])/(1-clip(P_stack[:,j]))) for j in range(P_stack.shape[1])]])
mu=Zs.mean(0); sd=Zs.std(0); sd[sd==0]=1; lr=LogisticRegression(C=.5,max_iter=2000).fit((Zs-mu)/sd,y[stack_mask]); pred_stack=lr.predict_proba((Zs-mu)/sd)[:,1]; print('META validation AUC',roc_auc_score(y[stack_mask],pred_stack),'coef',lr.coef_[0],flush=True)
# final hierarchy on all preserved training
mods_full=train_hierarchy(C.loc[tr],'full')
P_test=target_probs(mods_full,C.loc[te],'test')
Zt=np.column_stack([np.log(clip(p_lgb_test)/(1-clip(p_lgb_test))), *[np.log(clip(P_test[:,j])/(1-clip(P_test[:,j]))) for j in range(P_test.shape[1])]])
# handle if hierarchy depth counts differ: use common depth and refit meta on corresponding columns
common=min(P_stack.shape[1],P_test.shape[1]); cols=[0]+list(range(1,common+1)); Zs2=Zs[:,cols]; Zt2=Zt[:,cols]; mu2=Zs2.mean(0); sd2=Zs2.std(0); sd2[sd2==0]=1; lr2=LogisticRegression(C=.5,max_iter=2000).fit((Zs2-mu2)/sd2,y[stack_mask]); p_final=lr2.predict_proba((Zt2-mu2)/sd2)[:,1]
auc_lgb=roc_auc_score(y[te],p_lgb_test); aucs=[roc_auc_score(y[te],P_test[:,j]) for j in range(P_test.shape[1])]; auc_final=roc_auc_score(y[te],p_final)
print('FINAL LGBM',auc_lgb,'LSM',aucs,'STACK',auc_final,'coef',lr2.coef_[0],flush=True)
res={'lgbm_auc':auc_lgb,'lsm_depth_aucs':aucs,'stack_auc':auc_final,'stack_validation_auc':float(roc_auc_score(y[stack_mask],lr2.predict_proba((Zs2-mu2)/sd2)[:,1])),'coef':lr2.coef_[0].tolist(),'intercept':lr2.intercept_.tolist(),'n_features_lgbm':len(feats),'n_features_lsm':len(sel),'common_depths':common}
json.dump(res,open(OUT/'final_results.json','w'),indent=2)
pd.DataFrame({'row_index':np.flatnonzero(te),'y':y[te],'lgbm':p_lgb_test,'stack':p_final,**{f'lsm_d{j}':P_test[:,j] for j in range(P_test.shape[1])}}).to_csv(OUT/'test_scores.csv',index=False)
