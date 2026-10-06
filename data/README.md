# Data location

No patient-level data are committed to this public repository.

The analysis source data are stored in the private ZED Dropbox workspace at:

`/ZED/Research/UR_JOEL/anonymized_output/uk_peds/`

The final leakage-safe event-level modeling table used for the reported analysis is:

`pyuria_longitudinal_safe.csv`

The preserved patient-level split uses seed `20261006` and is defined by the split artifacts retained in the same private Dropbox folder. The original source extracts in that folder include encounter, diagnosis, problem-list, outpatient prescription, imaging, medication-administration, procedure, and laboratory files.

## Data governance

This repository intentionally excludes all row-level clinical data, identifiers, patient split manifests, and individual prediction files. Aggregate ROC, calibration, operating-characteristic, and feature-importance outputs are included under `results/`.
