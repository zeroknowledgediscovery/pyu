import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
B=Path('/mnt/data/joel_ua')
df=pd.read_csv(B/'pyuria_longitudinal_full.csv',low_memory=False)
sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv')
trp=set(pd.to_numeric(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'],errors='coerce').dropna().astype(int))
mrn=pd.to_numeric(df.Anon_MRN,errors='coerce')
train=mrn.isin(trp).to_numpy(); test=~train
y=df.pyuria_gt5.astype(int).to_numpy()
exclude={'Anon_Encounter_ID','Anon_MRN','UA_DTM','target_time','target_diff_min','target_wbc_raw','pyuria_gt5','ZIP_CD'}
unsafe_prefix=('prior_enc_','dx_','problem_','rx_')
features=[c for c in df.columns if c not in exclude and not c.startswith(unsafe_prefix)]
X=df[features].copy()
cat=[]
for c in features:
    if X[c].dtype=='object':
        X[c]=X[c].fillna('__MISSING__').astype('category'); cat.append(c)
    else:
        X[c]=pd.to_numeric(X[c],errors='coerce')
model=lgb.LGBMClassifier(n_estimators=500,learning_rate=.03,num_leaves=31,max_depth=-1,min_child_samples=30,subsample=.85,colsample_bytree=.8,reg_lambda=1.0,reg_alpha=.1,random_state=20261006,n_jobs=8,verbosity=-1)
model.fit(X.loc[train],y[train],categorical_feature=cat)
p=model.predict_proba(X.loc[test])[:,1]; auc=roc_auc_score(y[test],p)
print('features',len(features),'train',train.sum(),'test',test.sum(),'auc',auc)
imp=pd.DataFrame({'feature':features,'gain':model.booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False)
imp.to_csv(B/'safe_rich_lgbm_importance.csv',index=False)
pd.DataFrame({'row_index':np.flatnonzero(test),'y':y[test],'lgbm_prob':p}).to_csv(B/'safe_rich_lgbm_test_scores.csv',index=False)
(B/'safe_rich_lgbm_meta.json').write_text(json.dumps({'auc':auc,'n_features':len(features),'features':features,'categorical':cat},indent=2))
print(imp.head(25).to_string(index=False))
