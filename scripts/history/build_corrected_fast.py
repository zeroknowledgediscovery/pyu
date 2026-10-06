import pandas as pd, numpy as np, re, json, collections, warnings
from pathlib import Path
from openpyxl import load_workbook
warnings.filterwarnings('ignore')
B=Path('/mnt/data/joel_ua'); OUT=B/'pyuria_longitudinal_corrected.csv'
F=pd.read_csv(B/'pyuria_longitudinal_full.csv',low_memory=False)
F['UA_DTM']=pd.to_datetime(F.UA_DTM,errors='coerce'); F['Anon_MRN']=pd.to_numeric(F.Anon_MRN,errors='coerce').astype('Int64'); F['Anon_Encounter_ID']=pd.to_numeric(F.Anon_Encounter_ID,errors='coerce').astype('Int64'); F['_row']=np.arange(len(F))
sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv'); train=set(pd.to_numeric(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'],errors='coerce').dropna().astype(int)); allm=set(F.Anon_MRN.dropna().astype(int))
left=F[['_row','Anon_MRN','Anon_Encounter_ID','UA_DTM']].dropna(subset=['Anon_MRN','UA_DTM']).copy(); left['Anon_MRN']=left.Anon_MRN.astype(int)

def xlsx(path, cols):
    wb=load_workbook(path,read_only=True,data_only=True); ws=wb[wb.sheetnames[0]]; it=ws.iter_rows(values_only=True); hdr=list(next(it)); ids=[hdr.index(c) for c in cols]
    return pd.DataFrame((tuple(r[i] for i in ids) for r in it),columns=cols)
def safe(s): return re.sub(r'[^A-Za-z0-9]+','_',str(s)).strip('_')[:28]

def cum_at(h, queries, timecol='event_dt'):
    if h.empty: return np.zeros(len(F),dtype=np.int32)
    z=h[['Anon_MRN',timecol]].dropna().copy(); z['Anon_MRN']=pd.to_numeric(z.Anon_MRN,errors='coerce'); z=z[z.Anon_MRN.notna() & z.Anon_MRN.isin(allm)]; z['Anon_MRN']=z.Anon_MRN.astype('int64'); z=z.sort_values([timecol,'Anon_MRN']); z['cum']=z.groupby('Anon_MRN').cumcount()+1
    l=queries[['_row','Anon_MRN','qtime']].dropna(subset=['Anon_MRN','qtime']).copy(); l['Anon_MRN']=l.Anon_MRN.astype('int64'); l=l.sort_values(['qtime','Anon_MRN'])
    r=pd.merge_asof(l,z[['Anon_MRN',timecol,'cum']],left_on='qtime',right_on=timecol,by='Anon_MRN',direction='backward',allow_exact_matches=False)
    out=np.zeros(len(F),dtype=np.int32); out[r._row.to_numpy(int)]=r['cum'].fillna(0).to_numpy(np.int32); return out

def counts(prefix,h,timecol,windows=(30,365,None), qshift_hours=0, subtract_current=None):
    hh=h[['Anon_MRN',timecol]].rename(columns={timecol:'event_dt'}).copy()
    q0=left.copy(); q0['qtime']=q0.UA_DTM-pd.to_timedelta(qshift_hours,unit='h')
    life=cum_at(hh,q0)
    if subtract_current is not None: life=np.maximum(0,life-subtract_current)
    for w in windows:
        if w is None: F[prefix+'_life']=life
        else:
            qp=q0.copy(); qp['qtime']=qp.qtime-pd.to_timedelta(w,unit='D'); past=cum_at(hh,qp); v=life-past
            # current subtraction already included in life; past cannot contain current encounter
            F[f'{prefix}_{w}d']=np.maximum(0,v)

def current_enc_counts(h, enc_col='Anon_Encounter_ID'):
    cnt=h.dropna(subset=[enc_col]).groupby(enc_col).size()
    return F[enc_col].map(cnt).fillna(0).astype(np.int32).to_numpy()

def current_enc_cat_counts(h, cat, enc_col='Anon_Encounter_ID'):
    cnt=h[h[cat].notna()].groupby([enc_col,cat]).size()
    # specialized function returned per category later
    return cnt

print('encounter',flush=True)
enc=xlsx(B/'anonymized_PHI_UA_ENCOUNTER_MRNREFORMAT.xlsx',['Anon_Encounter_ID','Anon_MRN','ADMT_DT']); enc['event_dt']=pd.to_datetime(enc.ADMT_DT,errors='coerce'); enc['Anon_Encounter_ID']=pd.to_numeric(enc.Anon_Encounter_ID,errors='coerce').astype('Int64'); enc['Anon_MRN']=pd.to_numeric(enc.Anon_MRN,errors='coerce').astype('Int64')
counts('prior_enc',enc,'event_dt',(7,30,365,None),subtract_current=np.ones(len(F),dtype=np.int32))

print('dx',flush=True)
dx=xlsx(B/'anonymized_PHI_UA_DIAGNOSIS_MRNREFORMAT.xlsx',['Anon_Encounter_ID','Anon_MRN','ADMT_DT','DIAGNOSIS']); dx['event_dt']=pd.to_datetime(dx.ADMT_DT,errors='coerce'); dx['Anon_Encounter_ID']=pd.to_numeric(dx.Anon_Encounter_ID,errors='coerce').astype('Int64'); dx['Anon_MRN']=pd.to_numeric(dx.Anon_MRN,errors='coerce').astype('Int64'); dx['icd3']=dx.DIAGNOSIS.fillna('').astype(str).str.upper().str.replace('.','',regex=False).str[:3]
sub=current_enc_counts(dx); counts('dx',dx,'event_dt',(30,365,None),subtract_current=sub)
for nm,pat in {'uti':r'^N39','pyelo':r'^N1[0-2]','urinary':r'^R3','renal':r'^N[0-2]','fever':r'^R50','abd':r'^R10'}.items():
    z=dx[dx.icd3.str.match(pat,na=False)]; counts('dx_'+nm,z,'event_dt',(30,365,None),subtract_current=current_enc_counts(z))
topdx=[]

print('problem',flush=True)
pr=xlsx(B/'anonymized_PHI_UA_PROBLEM_LIST_MRNREFORMAT.xlsx',['Anon_Encounter_ID','Anon_MRN','PROBLEM_ICD10_LIST','CHRONIC_YN','NOTED_DATE','DATE_OF_ENTRY']); pr['event_dt']=pd.to_datetime(pr.DATE_OF_ENTRY,errors='coerce').fillna(pd.to_datetime(pr.NOTED_DATE,errors='coerce')); pr['Anon_Encounter_ID']=pd.to_numeric(pr.Anon_Encounter_ID,errors='coerce').astype('Int64'); pr['Anon_MRN']=pd.to_numeric(pr.Anon_MRN,errors='coerce').astype('Int64'); pr['icd3']=pr.PROBLEM_ICD10_LIST.fillna('').astype(str).str.split(',').str[0].str.upper().str.replace('.','',regex=False).str[:3]
counts('problem',pr,'event_dt',(365,None),subtract_current=current_enc_counts(pr)); zc=pr[pr.CHRONIC_YN.astype(str).str.upper().eq('Y')]; counts('problem_chronic',zc,'event_dt',(365,None),subtract_current=current_enc_counts(zc))
toppr=[]

print('rx',flush=True)
rx=xlsx(B/'anonymized_PHI_UA_OUTPATIENT_RX_MRNREFORMAT.xlsx',['Anon_MRN','ORDER_START_DATE','DISPLAY_NAME','DESCRIPTION']); rx['event_dt']=pd.to_datetime(rx.ORDER_START_DATE,errors='coerce'); rx['Anon_MRN']=pd.to_numeric(rx.Anon_MRN,errors='coerce').astype('Int64'); txt=(rx.DISPLAY_NAME.fillna('').astype(str)+' '+rx.DESCRIPTION.fillna('').astype(str)).str.lower(); rx['drug']=txt.str.extract(r'([a-z][a-z0-9-]{2,})',expand=False).fillna(''); abxpat=r'amoxic|ampic|penicill|cephal|cef[a-z]|sulfameth|trimeth|nitrofur|ciprof|levof|azith|doxy|clinda|vancom|gentamic|tobram|meropen|piperac|linezolid|metronid'; rx['abx']=txt.str.contains(abxpat,regex=True,na=False)
counts('rx',rx,'event_dt',(30,365,None),qshift_hours=24); counts('rx_abx',rx[rx.abx],'event_dt',(30,365,None),qshift_hours=24)
toprx=[]

topmed=[]; topproc=[]
F.drop(columns=['_row'],inplace=True); F=F.replace([np.inf,-np.inf],np.nan); F.to_csv(OUT,index=False)
meta={'shape':F.shape,'topdx':topdx,'toppr':toppr,'toprx':toprx,'topmed':topmed,'topproc':topproc,'timing_rules':{'diagnosis':'current encounter subtracted','problem':'current encounter subtracted','rx':'24h safety gap','encounter':'current encounter subtracted','medadmin':'exact timestamp','procedure':'service date +1 day'}}
(B/'pyuria_longitudinal_corrected_metadata.json').write_text(json.dumps(meta,indent=2)); print('DONE',F.shape,flush=True)
