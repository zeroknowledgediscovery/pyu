# pyu: leakage-safe pyuria prediction with LightGBM and multiscale LSMs

This repository contains the complete analysis code, fitted models, aggregate evaluation outputs, and reports for prediction of microscopic pyuria from pre-urinalysis pediatric EHR information.

The binary target is **urine WBC >5/HPF** (`pyuria_gt5`). Each observation is a urinalysis event. All predictive information is censored to information available before the urinalysis index time; channels whose timestamps could not reliably distinguish pre- from post-UA information were excluded from the final safe analysis.

## Main result

The final model stacks a conventional LightGBM predictor with four target-conditionals from a persistence-peeled sequence of Large Science Models (LSMs). The preserved patient-held-out test set contains 7,417 UA events (2,589 positive; prevalence 34.9%).

- LightGBM AUC: **0.7283**
- Multiscale LSM-only stack AUC: **0.7023**
- LightGBM + four multiscale LSM conditionals: **0.7304**
- zedstat smoothed upper-hull AUC: **0.7329** (95% CI 0.7204-0.7454)
- Brier score: **0.1934** (95% CI 0.1894-0.1974)
- Calibration intercept: **0.0426** (95% CI -0.0197-0.1053)
- Calibration slope: **1.064** (95% CI 0.999-1.134)

See [`report.md`](report.md) for the full analysis and [`results/brief.pdf`](results/brief.pdf) for a compact performance brief.

## Repository layout

```text
pyu/
├── README.md
├── report.md
├── requirements.txt
├── data/
│   └── README.md                 # private data location; no data committed
├── scripts/
│   ├── make_publication_figures.py
│   └── history/                  # scripts used during the full analysis campaign
├── models/
│   ├── lightgbm_final.txt
│   ├── lightgbm_final_meta.json
│   ├── lsm_depth_00/             # runtime binary trees + source maps
│   ├── lsm_depth_01/
│   ├── lsm_depth_02/
│   ├── lsm_depth_03/
│   ├── lsm_hierarchy.json
│   ├── lsm_feature_metadata.json
│   └── stack_logistic.json
└── results/
    ├── brief.tex
    ├── brief.pdf
    ├── final_performance.json
    ├── figures/
    ├── tables/
    └── zedstat/
```

## Data

No patient-level source data, identifiers, split manifests, or individual-level predictions are committed to this public repository. The private source data and final modeling table are stored in the ZED Dropbox workspace; see [`data/README.md`](data/README.md).

The final leakage-safe modeling table is `pyuria_longitudinal_safe.csv` in the private Dropbox project directory.

## Leakage control

A preliminary rich model reached AUC ~0.817, but it was rejected because current-encounter diagnosis/problem-list/outpatient-Rx timestamps could encode downstream information after the UA was ordered or resulted. The final reported model excludes these ambiguous channels and retains only reliably pre-UA information: demographics, anthropometrics, pre-UA vitals, precisely timestamped laboratory results, medication administrations, imaging, safely shifted procedure history, and prior-UA/pyuria history.

## Modeling

The preserved patient-level split uses seed `20261006` and is performed by `Anon_MRN`, preventing the same patient from appearing in both train and test.

### LightGBM

The final LightGBM uses 110 leakage-safe features. The serialized booster is `models/lightgbm_final.txt` and reproduces held-out AUC 0.728282.

### Multiscale LSM sequence

A 60-feature categorical LSM representation was learned with `alpha=0.1`. Robust persistence peeling generated residual populations:

```text
11,297 -> 3,521 -> 1,124 -> 376
```

At each depth, a separate LSM was fit. The target was masked at scoring time and the conditional probability `P(pyuria_gt5=1 | X_without_target)` was extracted from each model.

### Stacking

Patient-grouped out-of-fold predictions were used to train the logistic stack. The final stack uses LightGBM plus all four LSM target-conditionals; test labels were not used to fit the stack.

## zedstat evaluation

All final ROC processing, confidence bounds, operating characteristics, and calibration summaries were produced with the group's public `zedstat` package. Aggregate outputs are under `results/zedstat/`.

At the zedstat smoothed Youden operating point (threshold ~0.3175):

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Sensitivity | 0.773 | 0.756-0.788 |
| Specificity | 0.586 | 0.572-0.600 |
| PPV | 0.500 | 0.485-0.516 |
| NPV | 0.828 | 0.814-0.841 |
| Accuracy | 0.651 | 0.636-0.666 |
| LR+ | 1.87 | 1.77-1.97 |
| LR- | 0.388 | 0.353-0.426 |

High-specificity operating points are included in `results/zedstat/selected_high_specificity_operating_points.csv`.

## Most important variables

Held-out SHAP magnitude and LightGBM gain agree on several dominant predictors. The leading variables are age, gender, weight, prior pyuria rate, absolute neutrophil count, last prior urine-WBC value, CBC WBC, pre-UA laboratory burden, albumin, creatinine, pulse, and admission-to-UA time. Full ranked tables are under `results/tables/`.

## Reproducing figures

With the aggregate result tables already present:

```bash
python scripts/make_publication_figures.py
```

For full model reconstruction, place the private data at the location described in `data/README.md`, install `requirements.txt`, and use the scripts retained under `scripts/history/` as the exact analysis record.

## Notes

This is a research analysis, not a deployed clinical decision-support system. Performance should be externally validated and workflow-specific operating thresholds should be selected before clinical use.