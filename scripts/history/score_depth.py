import sys,json,time,argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/mnt/data/joel_mc/runtime');import predict_distribution as p
ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['stack','test'],required=True);ap.add_argument('--depth',type=int,required=True);args=ap.parse_args()
B=Path('/mnt/data/joel_ua');O=B/'final_multiscale';D=pd.read_csv(B/'pyuria_longitudinal_full.csv',low_memory=False);y=D.pyuria_gt5.astype(int).to_numpy();mrn=D.Anon_MRN.astype(int).to_numpy();sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv');train=set(pd.to_numeric(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'],errors='coerce').dropna().astype(int));rng=np.random.default_rng(20261007);u=np.array(sorted(train));perm=rng.permutation(u);nbase=int(np.floor(.8*len(u)));stackset=set(perm[nbase:]);mask=np.array([m in stackset for m in mrn]) if args.kind=='stack' else np.array([m not in train for m in mrn])
tr=json.load(open(O/'lsm_transform.json'));C=pd.DataFrame(index=D.index);C['pyuria_gt5']=D.pyuria_gt5.astype(int).astype(str)
for c in tr['features']:
 s=D[c];spec=tr['transform'][c]
 if spec['type']=='cat':v=s.fillna('').astype(str);C[c]=v.where(v.isin(spec['top']),'OTHER')
 elif spec['type']=='raw':C[c]=pd.to_numeric(s,errors='coerce').map(lambda v:'<NA>' if pd.isna(v) else str(v))
 else:
  e=np.array(spec['edges'],float); labs=[f'Q{i+1}' for i in range(len(e)-1)];C[c]=pd.cut(pd.to_numeric(s,errors='coerce'),bins=e,labels=labs,include_lowest=True).astype(str).replace('nan','<NA>')
A=C.loc[mask].fillna('<NA>').astype(str).to_numpy();A[:,0]='';m=O/('internal' if args.kind=='stack' else 'full')/f'model_{args.depth:02d}';vals=np.empty(len(A));t=time.time()
for i,row in enumerate(A): vals[i]=float(p.predict_distribution(str(m/'trees'/'binary'),0,row,raw=True,run_dir=str(m)).get('1',0.0))
np.save(O/f'P_{args.kind}_d{args.depth}.npy',vals);print(args.kind,args.depth,'n',len(vals),'sec',time.time()-t,'auc',roc_auc_score(y[mask],vals),flush=True)
