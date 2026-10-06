import sys,json,subprocess,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data/joel_mc/runtime'); import predict_distribution as p
B=Path('/mnt/data/joel_ua');O=B/'final_multiscale';D=pd.read_csv(B/'pyuria_longitudinal_full.csv',low_memory=False);mrn=D.Anon_MRN.astype(int).values;sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv');train=set(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'].astype(int));mask=np.array([m in train for m in mrn]);tr=json.load(open(O/'lsm_transform.json'));C=pd.DataFrame(index=D.index);C['pyuria_gt5']=D.pyuria_gt5.astype(int).astype(str)
for c in tr['features']:
 s=D[c];spec=tr['transform'][c]
 if spec['type']=='cat':v=s.fillna('').astype(str);C[c]=v.where(v.isin(spec['top']),'OTHER')
 elif spec['type']=='raw':C[c]=pd.to_numeric(s,errors='coerce').map(lambda v:'<NA>' if pd.isna(v) else str(v))
 else:
  e=np.array(spec['edges'],float);labs=[f'Q{i+1}' for i in range(len(e)-1)];C[c]=pd.cut(pd.to_numeric(s,errors='coerce'),bins=e,labels=labs,include_lowest=True).astype(str).replace('nan','<NA>')
current=C.loc[mask].fillna('<NA>').astype(str).reset_index(drop=True);root=O/'full';shutil.rmtree(root,ignore_errors=True);root.mkdir();BIN='/mnt/data/joel_mc/runtime/LSM_static_dev-static';hist=[]
for d in range(4):
 csv=root/f'depth_{d:02d}.csv';current.to_csv(csv,index=False);md=root/f'model_{d:02d}';t=time.time();subprocess.run([BIN,str(csv),'0.1',str(md),'--threads','5','--subset-mode','exact'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT);print('trained',d,len(current),time.time()-t,flush=True)
 trees=sorted(int(x.stem.split('_')[1]) for x in (md/'trees'/'binary').glob('tree_*.bin'));r=p.persistence_batch(str(md/'trees'/'binary'),current.to_numpy(dtype=str),run_dir=str(md),tree_ids=trees,threads=5,prob_floor=1e-12);pp=np.asarray(r['persistence'],float);used=np.asarray(r['n_used'],float);avg=np.divide(pp,used,out=np.full(len(pp),np.nan),where=used>0);med=np.nanmedian(avg);scale=1.4826*np.nanmedian(np.abs(avg-med));z=(med-avg)/scale if scale>0 else np.full(len(avg),np.nan);idx=np.flatnonzero(np.isfinite(z)&(z>.5));hist.append({'depth':d,'n':len(current),'next':len(idx),'median':float(med),'scale':float(scale)});print('next',len(idx),flush=True)
 if d<3:current=current.iloc[idx].reset_index(drop=True)
json.dump(hist,open(root/'hierarchy.json','w'),indent=2)
