# FLA-SH: Fast, Heterogeneous Dataset QuaLity Assessment via SHapley Values
A Chunked Data Shapley approximation where estimates Shapley values per chunk by implementing a chunk partitioning and a sampling permutation we prioritise to select the most probably high quality chunks during the subset selection.

### Dependencies:
1. `Python` version 3.9.21
2. `numpy=1.22.0 `
3. `pandas=1.3.5`
4. `pytorch=2.4.0`
5. `scikit-learn=1.6.1`
6. `scipy=1.8.0`
7. `tqdm=4.67.1`
8. `transformers=4.57.1`
9. `cv2-contrib=4.12`
10. `hilbertcurve=2.0.5`

### Run Data-Shapley example

For tabular data:
```
DS = Data_Shapley_Dataset(datasets_path='/path/to/csv/dir/',
                          outpath='Output/results/',
                          metric='Accuracy',
                          problem='classification',
                          model_type='MLP_Class')

Shap_values_dict, shap_values_average = DS.execute_per_dataset(split_data=True, chunk_size=256, split_by='cluster', data_type='tabular', k=50, use_hilbert_kernel=False)
```

#### Split Options

##### `split_by`
- `cluster` – K-Means clustering
- `trees` – KD-Trees
- `ordered` – ordering-based split

##### Time-Series Datasets
- `split_by='date'` or `'month'` splits data chronologically by day or month, respectively.

#### `data_type`
- `image`
- `text`
- `vectors` (when vector embeddings are provided)

#### Permutation Method
- `hilbert_kernel=False` → unified ratio sampling permutation
- `hilbert_kernel=True` → Hilbert kernel permutation
