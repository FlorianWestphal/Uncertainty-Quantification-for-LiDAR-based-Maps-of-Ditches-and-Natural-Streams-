# Raw Data

## Classification Performance

The classification performance results can be found in the folder  [performance](performance/). The CSV files under this folder list the performance metrics for each image. The last line report ts the average result over all images of that fold.

Those CSV files are organized depending on the DEM ([0.5 m](performance/05m/)
or [1 m](performance/1m/) and depending on the test fold, which was used to
generate these results.

## Uncertainty Quantification Performance

The uncertainty quantification performance results can be found in the folder
[uncertainty](uncertainty/). This folder contains the results, as produced by
the script
[evaluate_uncertainty_maps.py](../src/tools/evaluate_uncertainty_maps.py).

The result files are organized based on DEM resolution and used test fold as
the classification performance data.
It should be noted that the results for conformal regression (CR) are incorrect
in the files under [incorrect_cr](uncertainty/incorrect_cr/). However, all other
results are correct. The correct results for CR can be found in the folder
[correct_cr](uncertainty/correct_cr/). The results reported in the paper are
labeled `cr_cond`. The other results labeled `cr` are obtained when only one
list of non-conformity scores per class is used.
