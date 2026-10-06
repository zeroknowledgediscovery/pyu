# Analysis scripts

`history/` contains the scripts used during the analysis campaign, including dataset construction, leakage audit, LightGBM fitting, multiscale LSM fitting/scoring, patient-grouped OOF stacking, and zedstat evaluation.

`make_publication_figures.py` regenerates the public aggregate figures from the versioned result tables. It does not require patient-level data.

The scripts in `history/` retain the original working paths under `/mnt/data` and the private Dropbox project layout. They are kept as an exact analysis record rather than polished standalone command-line tools.
