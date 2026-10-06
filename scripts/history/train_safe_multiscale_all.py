import pandas as pd, numpy as np, json, os, sys, subprocess, shutil, time
from pathlib import Path
B=Path('/mnt/data/joel_ua'); OUT=B/'safe_multiscale_all'; OUT.mkdir(exist_ok=True)
DF=pd.read_csv(B/'pyuria_longitudinal_full.csv',low_memory=False)
sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv')
train_pat=set(pd.to_numeric(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'],errors='coerce').dropna().astype(int))
mrn=pd.to_numeric(DF.Anon_MRN,errors='coerce'); train_mask=mrn.isin(train_pat).to_numpy(); test_mask=~train_mask
imp=pd.read_csv(B/'safe_rich_lgbm_importance.csv')
# Top training-derived features; exclude suspect prefixes even if accidental.
unsafe=('prior_enc_','dx_','problem_','rx_')
features=[c for c in imp.feature if c in DF.columns and not c.startswith(unsafe)][:110]
# categorical representation learned from training only
Xcat=pd.DataFrame(index=DF.index)
meta={}
for c in features:
    s=DF[c]; tr=s[train_mask]
    if s.dtype=='object':
        vals=tr.fillna('').astype(str); vc=vals[vals.ne('')].value_counts(); keep=vc.head(24).index.tolist()
        def mapcat(x):
            x='' if pd.isna(x) else str(x)
            return x if (x=='' or x in keep) else '__OTHER__'
        Xcat[c]=s.map(mapcat)
        meta[c]={'type':'categorical','keep':keep}
    else:
        trn=pd.to_numeric(tr,errors='coerce'); sn=pd.to_numeric(s,errors='coerce')
        vals=trn.dropna().to_numpy(float); uniq=np.unique(vals)
        if len(uniq)<=12:
            # Preserve discrete counts/categories directly.
            Xcat[c]=sn.map(lambda x: '' if pd.isna(x) else str(float(x)).rstrip('0').rstrip('.') if isinstance(x,float) else str(x))
            meta[c]={'type':'discrete','states':int(len(uniq))}
        else:
            qs=np.linspace(0,1,9); edges=np.unique(np.quantile(vals,qs)) if len(vals) else np.array([])
            if len(edges)<3:
                Xcat[c]=sn.map(lambda x:'' if pd.isna(x) else str(x)); meta[c]={'type':'fallback'}
            else:
                edges[0]=-np.inf; edges[-1]=np.inf
                labels=[f'Q{i+1}' for i in range(len(edges)-1)]
                Xcat[c]=pd.cut(sn,bins=edges,labels=labels,include_lowest=True).astype('object').fillna('').astype(str)
                meta[c]={'type':'quantile','edges':[None if not np.isfinite(x) else float(x) for x in edges], 'labels':labels}
# Drop constant features on training
keep=[c for c in features if Xcat.loc[train_mask,c].nunique(dropna=False)>1]
Xcat=Xcat[keep]
out=pd.concat([DF[['pyuria_gt5']].astype(str),Xcat],axis=1)
train_df=out.loc[train_mask].reset_index(drop=True); test_df=out.loc[test_mask].reset_index(drop=True)
train_df.to_csv(OUT/'depth_00.csv',index=False); test_df.to_csv(OUT/'test.csv',index=False)
(Path(OUT/'feature_metadata.json')).write_text(json.dumps({'features':keep,'n_features':len(keep),'binning':{k:meta[k] for k in keep}},indent=2))
print('prepared',train_df.shape,test_df.shape,'features',len(keep),flush=True)
# Runtime setup
runtime=Path('/mnt/data/joel_mc/runtime'); sys.path.insert(0,str(runtime)); os.environ['LD_LIBRARY_PATH']=str(runtime)+':'+os.environ.get('LD_LIBRARY_PATH','')
import predict_distribution as pdist
LSM=Path('/mnt/data/joel_lsm_run/LSM_static_dev-static'); os.chmod(LSM,0o755)

def train_model(csv,model):
    if model.exists(): shutil.rmtree(model)
    cmd=[str(LSM),str(csv),'0.1',str(model),'--threads','8','--subset-mode','exact']
    print('TRAIN',' '.join(cmd),flush=True); t=time.time(); subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT); print('trained',model.name,'sec',time.time()-t,flush=True)

def tree_ids(model):
    return sorted(int(p.stem.split('_')[1]) for p in (model/'trees'/'binary').glob('tree_*.bin'))

def persistence(df,model):
    ids=tree_ids(model); r=pdist.persistence_batch(str(model/'trees'/'binary'),df.to_numpy(dtype=str),run_dir=str(model),tree_ids=ids,threads=8,prob_floor=1e-12)
    p=np.asarray(r['persistence'],float); n=np.asarray(r['n_used'],float); return np.divide(p,n,out=np.full_like(p,np.nan),where=n>0)

current=train_df
hier=[]
for d in range(4):
    csv=OUT/f'depth_{d:02d}.csv'; model=OUT/f'model_depth_{d:02d}'
    if d>0: current.to_csv(csv,index=False)
    train_model(csv,model)
    p=persistence(current,model); med=float(np.nanmedian(p)); mad=float(1.4826*np.nanmedian(np.abs(p-med))); z=(med-p)/mad if mad>0 else np.zeros(len(p)); sel=np.flatnonzero(np.isfinite(z)&(z>0.5))
    pd.DataFrame({'persistence':p,'lack_z':z,'selected':np.isin(np.arange(len(p)),sel).astype(int)}).to_csv(OUT/f'depth_{d:02d}_persistence.csv',index=False)
    hier.append({'depth':d,'n':len(current),'median_persistence':med,'mad_scale':mad,'next_n':int(len(sel))})
    print('depth',d,'n',len(current),'next',len(sel),flush=True)
    if d==3 or len(sel)<150: break
    current=current.iloc[sel].reset_index(drop=True)
(Path(OUT/'hierarchy.json')).write_text(json.dumps(hier,indent=2))
print('DONE',hier,flush=True)
