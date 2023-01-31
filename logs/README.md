# Overview

The data directory is organised in the following way:

| Path      | Role                                                                                                                                                                                   |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `models/` | The model parameters were stored here when running with the `--mode=test` flag. Here, the full model parameters are stored, as well as only the parameters of the classification head. |
| `test/`   | The training, validation, and test performance for `--mode=test` runs was stored here.                                                                                                 |
| `train/`  | The training and validation performance for `--mode=train` runs (the hyperparameter optimization) was stored here.                                                                     |
