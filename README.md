# FLA-SH: Fast, Dataset QuaLity and Fairness Assessment via SHapley Values
A Chunked Data Shapley approximation that estimates Shapley values per chunk. It combines a chunk partitioning step with a prioritised sampling permutation, so that the chunks most likely to be high quality are selected first during subset selection.

### Dependencies:
1. `Python` version 3.9.21
2. `numpy=1.22.0`
3. `pandas=1.3.5`
4. `pytorch=2.4.0`
5. `scikit-learn=1.6.1`
6. `scipy=1.8.0`
7. `tqdm=4.67.1`
8. `transformers=4.57.1`
9. `opencv-contrib-python=4.12`
10. `hilbertcurve=2.0.5`

### Run Data-Shapley example

For tabular data:
```python
from main import Data_Shapley_Dataset

DS = Data_Shapley_Dataset(datasets_path='/path/to/csv/dir/',
                          outpath='Output/results/',
                          metric='Accuracy',
                          problem='classification',
                          model_type='MLP_Class',
                          preprocessing='onehot_zscore',   # or 'legacy'
                          utility='accuracy')              # or 'neg_val_ce'

Shap_values_dict, shap_values_average = DS.execute_per_dataset(split_data=True, chunk_size=256,
                                                               split_by='cluster', data_type='tabular',
                                                               k=50, use_hilbert_kernel=False,
                                                               r_max=2, err=0.001)
```

#### Split Options

##### `split_by`
- `cluster` – K-Means clustering (on the raw features)
- `tree` – KD-Tree (on a PCA projection of the raw features)
- `ordered` – ordering-based split
- `random` – random split (image data only)

##### Time-Series Datasets
- `split_by='date'` or `'month'` splits data chronologically by day or month, respectively.

#### `data_type`
- `image`
- `tabular`
- `text`
- `vectors` (when vector embeddings are provided)

#### Permutation Options
- `use_hilbert_kernel=False` → unified ratio sampling (SR) permutation
- `use_hilbert_kernel=True` → Hilbert kernel permutation

#### Rounds
- `r_max` → round budget R_max (the paper uses `r_max=2`)
- `err` → truncation tolerance ε of Eq. 4. The rule is applied only from round 2 onwards and uses the unws M(Z ∪ {j}) − M(Z).

#### Model-input preprocessing (tabular)
- `preprocessing='onehot_zscore'` (default) → one-hot encodes the integer-coded categorical columns (`utils.CATEGORICAL_COLUMNS`) and z-scores all other columns. The transform is fitted on the training split only and applied unchanged to
validation and test data.
- `preprocessing='legacy'` → previous behaviour (per-row L2 normalisation)
- Partitioning and the SR/Hilbert priority orderings always use the raw features.
- Row-index columns (`Unnamed: 0`, blank headers) are dropped automatically when a CSV is loaded.

#### Coalition utility
- `utility='accuracy'` (default) → validation accuracy of the proxy model
- `utility='neg_val_ce'` → minus the mean validation cross-entropy (finer-grained; avoids ties)

### Run Fair-Flash example

```python
DS = Data_Shapley_Dataset(datasets_path='/path/to/csv/dir/',
                          outpath='Output/results/',
                          metric='Accuracy',
                          problem='classification',
                          model_type='MLP_Class',
                          use_fairness=True,
                          fairness_attr='gender',
                          lambda_fair=1.0,
                          fair_metric='groupdro')

Shap_values_dict, shap_values_average = DS.execute_per_dataset(split_data=True, chunk_size=256,
                                                               split_by='cluster', data_type='tabular',
                                                               k=50, use_hilbert_kernel=False, r_max=2)
```

#### Fairness settings
- `use_fairness=True` → enables fairness in the Shapley calculation
- `fairness_attr='gender'` → the dataset column whose subpopulation groups are examined for fairness
- `lambda_fair=1.0` → the weight of the fairness loss in the total Shapley contribution, in the range [0, 1]
- `fair_metric` → `'groupdro'` (Group-DRO, η = 0.1) or `'cvar'` (CVaR over the worst 30% of per-sample losses)

### Training notes
- Task-model retraining uses plain SGD, with early stopping on **validation loss** (patience 5). Set `model.early_stop_monitor = 'metric'` to stop on the validation metric instead.

### Reproducing the experiments
- Tabular datasets (Adult, Bank Marketing, Covertype, MIMIC-III):
  ```
  python run_vldb_tabular_experiment.py --chunk-size 256 --r-max 2
  ```
- Image datasets (CIFAR-10, MNIST, F-MNIST):
  ```
  python run_vldb_image_experiment.py --early-stopping <patience> --r-max 2
  ```
- Full-data baselines, with no valuation and no removal (train / val / test accuracy):
  ```
  python run_full_data_baseline.py
  python run_full_data_baseline_images.py --early-stopping <patience>
  ```
