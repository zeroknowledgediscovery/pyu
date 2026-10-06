# Complete analysis report: prediction of microscopic pyuria from pre-UA pediatric EHR data

## 1. Objective

The objective was to predict microscopic pyuria at the time a urinalysis (UA) is obtained, using information available before the urine microscopy result. The primary binary target is:

**`pyuria_gt5 = 1` if urine WBC >5/HPF; otherwise 0.**

The analysis was designed to compare a conventional high-performing tabular model (LightGBM) with Large Science Model (LSM) representations, including multiscale residual LSMs constructed by persistence peeling, and to determine whether LSM information adds orthogonal predictive value when stacked with LightGBM.

The final evaluation uses a fixed patient-level train/test split and a leakage-safe feature set. All final ROC, operating-characteristic, confidence-bound, and calibration calculations were performed with **zedstat**.

## 2. Data organization and index time

The unit of analysis is a **UA event**, not an encounter. For each UA event, the index time is the encounter's `UA_DTM`. Information used for prediction must be available before that time.

The complete private project data are stored at:

`/ZED/Research/UR_JOEL/anonymized_output/uk_peds/`

The final modeling table is:

`pyuria_longitudinal_safe.csv`

No patient-level data are committed to this repository.

### 2.1 Available source domains

The source package includes encounter, diagnosis, problem-list, outpatient medication, medication-administration, procedure, imaging, and laboratory extracts. The richer modeling campaign also derived prior-UA history and time-to-prior-UA features.

### 2.2 Leakage audit

An early rich model obtained LightGBM AUC ~0.817. That result was not accepted as the final performance estimate because current-encounter diagnosis, problem-list, outpatient-Rx, and related history fields can receive timestamps that do not reliably indicate whether the information existed before the UA. Such variables can encode the clinician's downstream response to the UA rather than baseline state.

The final safe model therefore excluded ambiguous diagnosis/problem-list/outpatient-Rx/encounter-history channels and retained channels with reliable pre-UA timing:

- demographics and anthropometrics;
- encounter-to-UA timing;
- pre-UA vital-sign extrema;
- precisely timestamped laboratory values and recency features;
- medication administrations before the UA;
- imaging history;
- procedure history using a conservative date safety shift;
- prior-UA count, prior pyuria count/rate, prior urine-WBC category, and recency of prior UA.

This produced **110 LightGBM predictors**.

## 3. Data split

The split is patient-level, based on `Anon_MRN`, so no patient appears in both train and test.

- random seed: **20261006**
- training fraction: **60% of unique patients**
- training UA events: **11,297**
- held-out test UA events: **7,417**
- test positives: **2,589**
- test negatives: **4,828**
- test prevalence: **34.91%**

The test set was preserved throughout model development. The logistic stacking model was fit from patient-grouped out-of-fold predictions generated within the training partition.

## 4. LightGBM model

The final LightGBM configuration was:

- 500 boosting rounds;
- learning rate 0.03;
- 31 leaves;
- minimum child samples 30;
- row subsampling 0.85;
- column subsampling 0.80;
- L2 regularization 1.0;
- L1 regularization 0.1;
- random seed 20261006.

The reconstructed serialized booster reproduces the recorded test AUC exactly:

**LightGBM AUC = 0.728282.**

The booster is stored at `models/lightgbm_final.txt`.

## 5. Multiscale LSM analysis

### 5.1 LSM feature representation

A 60-feature categorical representation was selected from the leakage-safe predictor set. Continuous variables were quantile-discretized using training-derived cut points; categorical variables were retained directly. The target was included as an LSM coordinate during training but **masked at prediction time**.

Each LSM was trained with:

**alpha = 0.1**.

### 5.2 Persistence peeling

Starting with the full training population, each depth used mean conditional log-persistence per usable coordinate. The next residual population retained the robust low-persistence tail using:

`z_lack = (median(P) - P(x)) / (1.4826 * MAD(P)) > 0.5`.

This generated the sequence:

| Depth | Training rows | Median persistence | Robust scale |
|---:|---:|---:|---:|
| 0 | 11,297 | -1.2799 | 0.2098 |
| 1 | 3,521 | -1.5232 | 0.1345 |
| 2 | 1,124 | -1.7177 | 0.1285 |
| 3 | 376 | -1.9488 | 0.1460 |

The four fitted models are stored as archives under `models/lsm_depth_00.tar.gz` through `lsm_depth_03.tar.gz`.

### 5.3 Target conditionals

For each sample and each depth, `pyuria_gt5` was masked and the LSM conditional probability

`P(pyuria_gt5 = 1 | X_without_target)`

was extracted. Individual residual models were informative but weaker than LightGBM; the root LSM was the strongest single LSM. The multiscale sequence was most useful as a set of complementary meta-features rather than as a hard persistence-based router.

## 6. Stacking

A logistic stack combined LightGBM probability with the four multiscale LSM target-conditionals. Meta-model fitting used patient-grouped out-of-fold training predictions only.

Held-out test AUCs:

| Model | Test AUC |
|---|---:|
| Multiscale LSMs only | 0.7023 |
| LightGBM | 0.7283 |
| LightGBM + root LSM | 0.7298 |
| LightGBM + depths 0-1 | 0.7300 |
| **LightGBM + depths 0-3** | **0.7304** |

The improvement over LightGBM alone is modest but directionally consistent: multiscale LSM conditionals contain information not fully captured by the conventional booster.

The final stack metadata are stored in `models/stack_logistic.json`.

## 7. zedstat performance analysis

The fixed held-out final stack scores were passed to zedstat without retraining.

### 7.1 ROC and AUC

- empirical held-out AUC: **0.7304**
- stratified-bootstrap 95% CI: **0.7184-0.7420**
- zedstat smoothed upper-hull AUC: **0.7329**
- hull 95% CI: **0.7204-0.7454**

The smoothed-hull result is shown in `results/figures/roc_curve_zedstat.png` and `.pdf`.

### 7.2 Youden operating point

zedstat identified a smoothed-hull Youden operating point at approximately threshold **0.3175**.

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Sensitivity | 0.773 | 0.756-0.788 |
| Specificity | 0.586 | 0.572-0.600 |
| PPV | 0.500 | 0.485-0.516 |
| NPV | 0.828 | 0.814-0.841 |
| Accuracy | 0.651 | 0.636-0.666 |
| LR+ | 1.866 | 1.767-1.970 |
| LR- | 0.388 | 0.353-0.426 |

### 7.3 High-specificity operating points

At more conservative thresholds:

| Target specificity | Sensitivity | PPV | LR+ |
|---:|---:|---:|---:|
| 0.90 | 0.316 | 0.629 | 3.16 |
| 0.95 | 0.213 | 0.696 | 4.27 |
| 0.99 | 0.0668 | 0.782 | 6.68 |

The complete operating-characteristic table is in `results/zedstat/zedstat_smoothed_hull_operating_metrics.csv`.

### 7.4 Calibration

Calibration of the final held-out score is favorable:

- Brier score: **0.1934** (95% CI 0.1894-0.1974)
- calibration intercept: **0.0426** (95% CI -0.0197-0.1053)
- calibration slope: **1.0643** (95% CI 0.9991-1.1336)

The ideal calibration values, intercept 0 and slope 1, lie within the reported intervals. The upper risk decile shows mild underprediction: mean predicted probability ~0.650 versus observed event rate ~0.702.

The zedstat reliability diagram is in `results/figures/calibration_curve_zedstat.*`.

## 8. Precision-recall behavior

At the observed prevalence of 34.9%, the zedstat-derived precision-recall curve shows the expected tradeoff between sensitivity and PPV. The precision baseline is the cohort prevalence. The figure is available at `results/figures/precision_recall_zedstat.*`.

## 9. Variables driving classification

Two complementary importance analyses were performed on the final leakage-safe LightGBM model:

1. **Gain importance**: how much tree-splitting gain is attributed to a feature during model construction.
2. **Held-out SHAP magnitude**: mean absolute SHAP contribution on the untouched test cohort, which summarizes how strongly each feature changes individual predictions.

The leading held-out SHAP features are:

| Rank | Feature | Mean |SHAP| | Interpretation |
|---:|---|---:|---|
| 1 | AGE | 0.312 | age-dependent baseline risk and presentation |
| 2 | GENDER | 0.306 | strong sex-associated difference in pyuria risk |
| 3 | WT_KG | 0.205 | age/body-size correlated clinical context |
| 4 | prior_pyuria_rate | 0.0947 | individual history of prior pyuria |
| 5 | lab_neutrophil_abs | 0.0894 | systemic inflammatory state |
| 6 | prior_ua_wbc_last | 0.0677 | most recent prior urinary WBC state |
| 7 | lab_cbc_wbc | 0.0636 | systemic leukocyte burden |
| 8 | preUA_lab_count | 0.0614 | intensity/complexity of pre-UA evaluation |
| 9 | lab_Albumin_Plasma | 0.0564 | systemic clinical state |
| 10 | lab_creatinine | 0.0512 | renal/clinical context |
| 11 | preUA_MAX_PULSE | 0.0462 | physiologic state |
| 12 | admission_to_ua_hours | 0.0451 | timing within the encounter |

The broad agreement between gain and SHAP is important: age, gender, weight, prior pyuria, prior urine-WBC state, CBC/inflammatory variables, and encounter physiology appear repeatedly among the strongest predictors.

These importance values are associative model explanations, not causal effects. Correlated variables can divide or exchange importance.

Full rankings are stored in:

- `results/tables/lightgbm_feature_importance_gain.csv`
- `results/tables/lightgbm_feature_importance_shap.csv`
- `results/tables/feature_importance_combined.csv`

## 10. Interpretation

The key result is that careful feature engineering and leakage control matter more than increasingly elaborate score transformations. Expanding from the original limited table to a richer, reliably pre-UA EHR representation raised LightGBM performance from roughly 0.705 to 0.728. The multiscale LSM sequence added a smaller but reproducible increment to 0.7304.

The LSM result is nevertheless scientifically useful. The four residual conditional fields contribute jointly; attempts to route each patient to a single residual model by persistence were inferior. This supports viewing the residual sequence as a multiscale representation whose conditional predictions can be combined rather than as mutually exclusive patient strata.

## 11. Limitations

- Single-institution retrospective analysis.
- The target is microscopic pyuria, not culture-confirmed UTI.
- Some potentially useful EHR channels were intentionally excluded because their timestamps could not guarantee pre-UA availability.
- The importance analysis is descriptive and non-causal.
- Model calibration and operating points require prospective and external validation before clinical deployment.
- Confidence intervals reported here condition on the fixed model-development procedure and do not incorporate uncertainty from model/feature selection.

## 12. Reproducibility artifacts

The repository contains:

- the final serialized LightGBM booster;
- all four fitted LSM residual models;
- stack coefficients and hierarchy metadata;
- scripts from the analysis campaign;
- aggregate zedstat ROC, hull, operating, calibration, and importance tables;
- publication-ready ROC, precision-recall, calibration, model-comparison, and feature-importance figures;
- `results/brief.tex` and compiled `results/brief.pdf`.

Patient-level source data remain private and are intentionally not versioned here.
