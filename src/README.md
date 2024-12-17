# Source Code

## Installation

Install the dependencies listed in [requirements.txt](requirements.txt).

In general, it should be sufficient to install
[auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA), pandas, and
matplotlib.


## Data Format

The provided scripts expect that the input and groundtruth data is provided as
`tif` files with a size of $500 \times 500$ pixels.
Corresponding input and groundtruth files should have the same name and should
be located in different folders. These folders should contain all images
(train, validation, calibration and test data). The selection, which images are
used for what is done by files containing one image name per line, for example,
`train.dat`:
```
18H007_73325_8825_25_0003
18H007_73325_8825_25_0004
18H007_73325_8825_25_0005
…
```

## Training

Models for semantic segmentation can be trained using the training script
[train.py](tools/train.py).

The main arguments to provide are: `-I`, `gt_path`, `log_path`, `train_ids`,
`valid_ids`.
The arguments `-I` and `gt_path` expect paths to the folder containing the
input and groundtruth images respectively.
The `log_path` specifies where the best performing model should be stored, as
well as logging data from the training run.
The images used for training and validation should be listed in the files
provided under `train_ids` and `valid_ids` respectively.

For all further arguments, please refer to the help of the training script.


## Creating Uncertainty Maps

After a model has been trained, the script
[map_uncertainty.py](tools/map_uncertainty.py) can be used to create the
uncertainty estimates using the different approaches for a given set of input
images.
Some of the methods (Feature Conformal Prediction (FCP) and Conformal
Regression (CR)) require an additional calibration set, which needs to be
provided.

### Execution

The main arguments to be provided are `-I`, `model_path`, `out_path`, and
`selected_ids`.
With `-I` the folder(s) to the input data can be provided, and the images for
which the model uncertainty should be quantified should be provided in the file
provided for the `selected_ids` argument (one file name per line, as above).
The uncertainty maps are stored in the directory provided as `out_path`.

Each of the uncertainty quantification methods needs to be enabled with its
respective switch (`--mc_dropout`, `--fcp`, `--cr`). Additionally, the path to
the groundtruth images (`--gt_path`) and the path to a file containing the
names of the calibration images (`--calibration_ids`) need to be provided for FCP and CR.

For all further arguments, please refer to the help of the uncertainty quantification script.

### Output

The script produces for all methods one PNG image, which can be used to
quickly view the uncertainty, as well as the raw uncertainty values saved in an
NPZ file.

Content of the NPZ files:
* `pred_{img_name}.npz`
    * `predicted` - the predicted probabilities, which can be used to infer the
      uncertainty based on network probability
* `mc_en_{img_name}.npz`
    * `uq` - uncertainty estimated by predictive entropy
* `mc_mi_{img_name}.npz`
    * `uq` - uncertainties estimated by mutual information
* `fcp_{img_name}.npz`
    * `uq` - uncertainties estimated by FCP
    * `intervals` - regression intervals for each pixel and each class
      (`class`, `width`, `height`, `bounds`) with lower bound at index 0 and
      upper bound at index 1
* `cr_cond_{img_name}.npz`
    * `uq` - uncertainties estimated by CR
    * `intervals` - regression intervals for each pixel and each class
      (`class`, `width`, `height`, `bounds`) with lower bound at index 0 and
      upper bound at index 1


## Model Evaluation

The performance of the trained model can be assessed with the script
[evaluate.py](tools/evaluate.py).
The main arguments to be provided are `-I`, `gt_path`, `model_path`, `out_path`, and
`selected_ids`.
With `-I` and `gt_path`, the folder(s) to the input and groundtruth data can be
provided, and the images which should be used for evaluation should be provided
in the file provided for the `selected_ids` argument (one file name per line,
as above). The results are stored in the CSV file whose path is given in
`out_path`.


## Uncertainty Evaluation


The performance of the trained model can be assessed with the script
[evaluate_uncertainty_maps.py](tools/evaluate_uncertainty_maps.py).

### Execution

The main arguments to be provided are `gt_path`, `map_path`, `out_path`, and
`selected_ids`, as well as `-u` and `-c`.

The `map_path` is the path to the folder where the uncertainty maps produced by
the `map_uncertainty.py` script are stored (The path provided as `out_path` to
the `map_uncertainty.py` script). The `selected_ids` argument specifies which
files should be used for evaluation (one file name per line, as above). The
`gt_path` and the `out_path` arguments are used to specify the location of the
groundtruth tiff files and the location where the evaluation results should be
stored.

Lastly, `-u` and `-c` are switches used to enable the evaluation of all
uncertainty quantification approaches and a particular evaluation of the
conformal prediction approaches respectively.

### Output

When the `-u` flag is set, the script outputs one `results.npz` file. This file
contains:
* `perf` - F1 scores and MCC values for each method and each step, where the
         most uncertain 1% of pixels are dropped at each step [UQ, steps, classes]
* `uncertainties` - mean and standard deviation of the uncertainty at each step
  [UQ, steps, mean/std]
* `auses` - table of the AUSE scores per class and in total [UQ, classes]
* `corr` - F1 scores and MCC values for each method and each step, where the
         most uncertain 1% of pixels are corrected at each step [UQ, steps, classes]
* `stream_corr` - F1 scores and MCC values for each method and each step, where the
                most uncertain 1% of pixels, which were predicted to be a
                stream are corrected at each step [UQ, steps, classes]
* `ditch_corr` - F1 scores and MCC values for each method and each step, where
               the most uncertain 1% of pixels, which were predicted to be a
               ditch are corrected at each step [UQ, steps, classes]
* `counts` - total number of pixels
* `auces` - table of the AUSE scores per class and in total [UQ, classes]

Furthermore, the script writes out a CSV file for each uncertainty
quantification method (`plot_{method}.csv`) with the following columns:
1) step number
2) MCC for UQ method
3) MCC for oracle
4) uncertainty mean
5) uncertainty std
6) correction: F1 background
7) correction: F1 ditch
8) correction: F1 stream
9) correction: MCC
10) correction: MCC for oracle
11) stream correction: F1 ditch
12) ditch correction: F1 stream

When the `-c` flag is set, the script outputs the following CSV files based on
the assessment of the conformal prediction based approaches:

* `uq_correctness.csv` - per class accuracy for each method, based on if the
  correct class is included in the prediction set
* `uq_bands.csv` - average interval size between the upper and lower bound
* `uq_cls_nums.csv` - average number of classes in the prediction set
* `uq_drop_perf.csv` - F1 score and MCC values when only instances are
  considered that have a single class prediction
* `uq_drop_drop.csv` - Recall for the single class predictions (compared to all
                     instances)
