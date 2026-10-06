from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

B=Path('/mnt/data/joel_ua')
df=pd.read_csv(B/'pyuria_longitudinal_safe.csv', low_memory=False)
sp=pd.read_csv('/mnt/data/joel_mc/split_patients_seed20261006.csv')
trp=set(pd.to_numeric(sp.loc[sp.split.astype(str).str.lower().eq('train'),'Anon_MRN'],errors='coerce').dropna().astype(int))
mrn=pd.to_numeric(df.Anon_MRN,errors='coerce')
train=mrn.isin(trp).to_numpy(); test=~train
y=df.pyuria_gt5.astype(int).to_numpy()
meta=json.loads((B/'safe_rich_lgbm_meta.json').read_text())
features=meta['features']
X=df[features].copy()
cat=[]
for c in features:
    if X[c].dtype=='object':
        X[c]=X[c].fillna('__MISSING__').astype('category'); cat.append(c)
    else:
        X[c]=pd.to_numeric(X[c],errors='coerce')
model=lgb.LGBMClassifier(n_estimators=500,learning_rate=.03,num_leaves=31,max_depth=-1,min_child_samples=30,subsample=.85,colsample_bytree=.8,reg_lambda=1.0,reg_alpha=.1,random_state=20261006,n_jobs=8,verbosity=-1)
model.fit(X.loc[train],y[train],categorical_feature=cat)
p=model.predict_proba(X.loc[test])[:,1]
auc=float(roc_auc_score(y[test],p))
print('AUC',auc)
if abs(auc-meta['auc'])>1e-10:
    raise SystemExit(f'AUC mismatch expected {meta["auc"]} got {auc}')
model.booster_.save_model(str(B/'final_lightgbm_model.txt'))
# gain + split
imp=pd.DataFrame({
    'feature':features,
    'gain':model.booster_.feature_importance(importance_type='gain'),
    'split':model.booster_.feature_importance(importance_type='split')
})
imp['gain_fraction']=imp['gain']/imp['gain'].sum()
imp=imp.sort_values('gain',ascending=False)
imp.to_csv(B/'final_lightgbm_importance.csv',index=False)
# SHAP contributions on untouched test set
contrib=model.booster_.predict(X.loc[test], pred_contrib=True)
shap=contrib[:,:-1]
shap_imp=pd.DataFrame({
    'feature':features,
    'mean_abs_shap':np.mean(np.abs(shap),axis=0),
    'mean_shap':np.mean(shap,axis=0),
    'sd_shap':np.std(shap,axis=0),
})
shap_imp['mean_abs_fraction']=shap_imp.mean_abs_shap/shap_imp.mean_abs_shap.sum()
shap_imp=shap_imp.sort_values('mean_abs_shap',ascending=False)
shap_imp.to_csv(B/'final_lightgbm_shap_importance.csv',index=False)
# Meta
(B/'final_lightgbm_model_meta.json').write_text(json.dumps({
    'auc':auc,'seed':20261006,'n_train':int(train.sum()),'n_test':int(test.sum()),
    'n_features':len(features),'features':features,'categorical':cat,
    'params':model.get_params()
},indent=2,default=str))
print(shap_imp.head(25).to_string(index=False))
