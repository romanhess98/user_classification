# Overview

The data directory is organised in the following way:

| Path        | Role                           |
|-------------|--------------------------------|
| `analysis/` | Some data used for analysis.   |
| `/`          | The datasets used for training |



# Datasets

The content of the datasets is described in the following table:

| Name                         | Content                                                                                                       |
|------------------------------|---------------------------------------------------------------------------------------------------------------|
| `df_raw.csv`                 | The raw full dataset.                                                                                         |
| `df_{event}.csv`             | All data available for the corresponding event.                                                               |
| `df_{event}_train.csv`       | The training data of the corresponding event.                                                                 |
| `df_{event}_val.csv`         | The validation data of the corresponding event.                                                               |
| `df_{event}_test.csv`        | The test data of the corresponding event.                                                                     |
| `df_{event}_nd_c.csv`        | The preprocessed data for the corresponding event (_nd_=no duplicates, _c_=consistent/no conflicting labels). |
| `df_{event}_nd_c_train.csv`  | The preprocessed training data for the corresponding event.                                                   |
| `df_{event}_nd_c_val.csv`    | The preprocessed validation data for the corresponding event.                                                 |
| `df_{event}_nd_c_test.csv`   | The preprocessed test data for the corresponding event.                                                       |

In this study, only the `df_{event}_nd_c_train.csv`, `df_{event}_nd_c_val.csv`, and `df_{event}_nd_c_test.csv` datasets were given to the model.
The others were mostly created for exploration and descriptive analyses.