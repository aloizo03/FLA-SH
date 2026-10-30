import os
import glob
import time
import random
import shutil
import argparse
import traceback

import numpy as np
import pandas as pd
from utils import drop_index_columns
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from main import Data_Shapley_Dataset
from utils_dataShapley import create_model
from utils import get_metric, transform_to_torch, FeatureTransform, PREPROCESSING_MODES

PROJECT_ROOT = os.environ.get(
    'DATA_SHAPLEY_ROOT',
    '/FLA-SH/')

HOLDOUT_ROOT = os.path.join(PROJECT_ROOT, 'Data', 'holdout_80_20')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results')
WORK_ROOT = os.path.join(PROJECT_ROOT, 'Output', 'results', 'tabular')

LABEL_CLASS = 'class'
HIDDEN_LAYERS = [128, 256, 512]
MODEL_TYPE = 'MLP_Class'
METRIC = 'Accuracy'
PROBLEM = 'classification'
PERCENTAGES = np.arange(0.1, 0.55, 0.1, dtype=np.float64)

CHUNK_SIZE = 250
K_SUBSETS = 50
SR_SPLIT_BY = 'ordered'

STANDARDIZE = True

PREPROCESSING = 'onehot_zscore'

R_MAX = 2

VARIANTS = {
    'FLA-SH-KM-SR': ('cluster', False),
    'FLA-SH-KDT-SR': ('tree', False),
    'FLA-SH-KM-HIL': ('cluster', True),
    'FLA-SH-KDT-HIL': ('tree', True),
    'FLA-SH-SR': (SR_SPLIT_BY, False),
    'FLA-SH-HIL': (SR_SPLIT_BY, True),
}

DATASETS = {
    'adult': {
        'source_csv': os.path.join(PROJECT_ROOT, 'Data', 'Adult', 'adult.csv'),
        'out_name': 'Adult',
        'num_of_classes': 2,
    },
    'bank_marketing': {
        'source_csv': os.path.join(PROJECT_ROOT, 'Data', 'Bank_Marketing', 'bank_marketing.csv'),
        'out_name': 'Bank Marketing',
        'num_of_classes': 2,
    },
    'covertype': {
        'source_csv': os.path.join(PROJECT_ROOT, 'Data', 'covertype_dataset', 'covertype_noisy.csv'),
        'out_name': 'covertype',
        'num_of_classes': 7,
    },
    'mimic': {
        'source_csv': os.path.join(PROJECT_ROOT, 'Data', 'mimic_no_leak', 'mimic.csv'),
        'out_name': 'Mimic',
        'num_of_classes': 2,
    },
}


def make_holdout_split(dataset_key, source_csv, seed, test_size=0.2):
    df = drop_index_columns(pd.read_csv(source_csv)).fillna(-99)
    y = df[LABEL_CLASS]
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=y)

    root = os.path.join(HOLDOUT_ROOT, dataset_key)
    train_dir = os.path.join(root, 'train80')
    test_csv = os.path.join(root, 'test20_HELD_OUT.csv')

    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir, exist_ok=True)
    train_path = os.path.join(train_dir, f'{dataset_key}_train80.csv')
    train_df.to_csv(train_path, index=False)

    os.makedirs(root, exist_ok=True)
    test_df.to_csv(test_csv, index=False)

    print(f'  [{dataset_key}] holdout split (seed={seed}): '
          f'{len(train_df)} train (80%) / {len(test_df)} test (20%)')
    print(f'    train dir (only thing Data_Shapley_Dataset reads): {train_dir}')
    print(f'    held-out test csv (never read by FLA-SH):          {test_csv}')
    return train_path, test_csv, train_dir, len(train_df), len(test_df)


def _read_chunk_xy(csv_path, label_class=LABEL_CLASS):
    """Read a chunk CSV and return (X, y); columns starting with 'class' are the labels."""
    df = drop_index_columns(pd.read_csv(csv_path, index_col=False)).fillna(-99)
    class_cols = df.loc[:, df.columns.str.startswith(label_class)]
    X = df.drop(columns=list(class_cols.columns))
    return X, class_cols


def _train_predict_accuracy(ds_obj, X_train, y_train, X_test, y_test,
                            num_classes, eval_epochs, eval_bs, eval_lr, return_pred=False,
                            return_split_acc=False):
    """Train the task model on (X_train, y_train) and return its test accuracy."""
    norm = not STANDARDIZE
    if PREPROCESSING == 'onehot_zscore':
        ft = FeatureTransform(X_train)
        X_train, X_test = ft(X_train), ft(X_test)
        norm = False
    elif STANDARDIZE:
        Xtr = np.asarray(X_train.values if hasattr(X_train, 'values') else X_train, dtype=float)
        Xte = np.asarray(X_test.values if hasattr(X_test, 'values') else X_test, dtype=float)
        scaler = StandardScaler().fit(Xtr)
        X_train = scaler.transform(Xtr)
        X_test = scaler.transform(Xte)

    model = create_model(model_type=ds_obj.model_type, problem=ds_obj.problem,
                         max_epochs=eval_epochs, batch_size=eval_bs, lr=eval_lr,
                         val_frac=0.1, hidden_layer_size=ds_obj.hidden_layers,
                         use_img=False, use_NN=True, num_classes=num_classes,
                         norm=norm)

    y_train_1d = y_train.values.ravel() if hasattr(y_train, 'values') else np.ravel(y_train)

    n_tr = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
    fit_split = None
    if n_tr >= 20 and len(np.unique(y_train_1d)) > 1:
        try:
            X_tr2, X_val, y_tr2, y_val = train_test_split(
                X_train, y_train_1d, test_size=0.1, stratify=y_train_1d, random_state=0)
            model.fit(X_tr2, y_tr2, X_val=X_val, y_val=y_val, use_val=True)
            fit_split = (X_tr2, X_val, y_tr2, y_val)
        except Exception:
            model.fit(X_train, y_train_1d, X_val=None, y_val=False, use_val=False)
    else:
        model.fit(X_train, y_train_1d, X_val=None, y_val=False, use_val=False)

    X_torch = transform_to_torch(X_test).to(model.device)
    y_pred = model.predict(X_torch).detach().cpu().numpy()
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = np.argmax(y_pred.reshape(1, -1), axis=1)

    inv_map = getattr(model, '_label_map_inv', None)
    if inv_map:
        y_pred = np.array([inv_map.get(int(p), int(p)) for p in y_pred])

    y_test_1d = y_test.values.ravel() if hasattr(y_test, 'values') else np.ravel(y_test)
    acc = get_metric(metric='accuracy', y_pred=y_pred, y=y_test_1d)
    if return_pred:
        return acc, y_pred
    if return_split_acc:
        def _acc(X, y):
            p = model.predict(transform_to_torch(X).to(model.device)).detach().cpu().numpy()
            p = np.argmax(p, axis=1) if p.ndim == 2 else np.argmax(p.reshape(1, -1), axis=1)
            if inv_map:
                p = np.array([inv_map.get(int(v), int(v)) for v in p])
            return get_metric(metric='accuracy', y_pred=p, y=np.ravel(y))

        if fit_split is None:
            return {'train': float('nan'), 'val': float('nan'), 'test': acc}
        X_tr2, X_val, y_tr2, y_val = fit_split
        return {'train': _acc(X_tr2, y_tr2), 'val': _acc(X_val, y_val), 'test': acc}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acc


def evaluate_data_selection_fixed_test(scores_dict, ds_obj, train_dir, X_test, y_test,
                                       percentages, eval_epochs, eval_bs, eval_lr, itter):
    train_dir_abs = os.path.abspath(train_dir)
    scores_sorted = {name: val for name, val in
                     sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)}
    chunk_names = list(scores_sorted.keys())
    n_chunks = len(chunk_names)

    mean_lst, std_lst = [], []
    for perc in percentages:
        n_remove = int(np.ceil(n_chunks * perc))
        n_selected = max(1, n_chunks - n_remove)
        selected = chunk_names[:n_selected]

        X_parts, y_parts = [], []
        for name in selected:
            fpath = ds_obj.datasets_filenames.get(name)
            if fpath is None:
                continue
            X_i, y_i = _read_chunk_xy(fpath, LABEL_CLASS)
            X_parts.append(X_i)
            y_parts.append(y_i)

        if not X_parts:
            print(f'    [warn] no chunk files resolved for lambda={perc:.2f}; skipping')
            mean_lst.append(np.nan)
            std_lst.append(np.nan)
            continue

        X_train = pd.concat(X_parts, axis=0)
        y_train = pd.concat(y_parts, axis=0)
        num_classes = int(max(ds_obj.num_of_classes or 0, len(np.unique(y_train.values.ravel()))))

        runs = []
        for _ in range(itter):
            acc = _train_predict_accuracy(ds_obj, X_train, y_train, X_test, y_test,
                                          num_classes, eval_epochs, eval_bs, eval_lr)
            runs.append(acc)

        runs = np.asarray(runs, dtype=np.float64)
        mean_lst.append(float(runs.mean()))
        std_lst.append(float(runs.std(ddof=1)) if len(runs) > 1 else 0.0)
        print(f'    lambda={perc:.2f}  selected {n_selected}/{n_chunks} train chunks  '
              f'acc(on FIXED test)={runs.mean():.4f} +/- {std_lst[-1]:.4f}  runs={np.round(runs, 4)}')

    return mean_lst, std_lst


def run_variant(dataset_key, cfg, variant_name, split_by, use_hilbert, train_dir,
                chunk_size, k, X_test, y_test, eval_epochs, eval_bs, eval_lr, itter, n_train):
    found = glob.glob(os.path.join(train_dir, '*.csv'))
    assert len(found) == 1, f"expected exactly 1 CSV in {train_dir}, found {found}"

    out_dir = os.path.join(WORK_ROOT, dataset_key, variant_name)
    ds = Data_Shapley_Dataset(
        datasets_path=train_dir,
        outpath=out_dir,
        metric=METRIC,
        problem=PROBLEM,
        model_type=MODEL_TYPE,
        num_of_classes=cfg['num_of_classes'],
        hidden_layers=list(HIDDEN_LAYERS),
        label_class=LABEL_CLASS,
        preprocessing=PREPROCESSING,
    )

    print(f'  [{variant_name}] computing FLA-SH values on the 80% TRAIN split only '
          f'(split_by={split_by}, hilbert={use_hilbert}, chunk_size={chunk_size}, k={k}) ...')
    t0 = time.perf_counter()
    out_shapley_values, _ = ds.execute_per_dataset(
        split_data=True, chunk_size=chunk_size, split_by=split_by,
        use_hilbert_kernel=use_hilbert, k=k, smart_selection=True, data_type='tabular',
        r_max=R_MAX)
    flash_seconds = time.perf_counter() - t0
    inner = list(out_shapley_values.values())[0]

    total_chunk_rows = sum(len(pd.read_csv(fp)) for fp in ds.datasets_filenames.values())
    assert total_chunk_rows == n_train, (
        f"LEAK/BUG: chunks cover {total_chunk_rows} rows, expected exactly n_train={n_train}.")

    print(f'  [{variant_name}] {len(inner)} train-only chunks scored '
          f'({total_chunk_rows} rows, verified == n_train={n_train}) in {flash_seconds:.2f}s; '
          f'evaluating against the FIXED, held-out test set ...')

    means, stds = evaluate_data_selection_fixed_test(
        scores_dict=inner, ds_obj=ds, train_dir=train_dir, X_test=X_test, y_test=y_test,
        percentages=PERCENTAGES, eval_epochs=eval_epochs, eval_bs=eval_bs,
        eval_lr=eval_lr, itter=itter)

    # Free GPU memory before the next variant.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return means, stds, flash_seconds


def _load_existing_results(out_dir, dataset_key, percentages):
    """Load saved results for a dataset so finished variants can be skipped."""
    acc_path = os.path.join(out_dir, f'accuracy_{dataset_key}.csv')
    std_path = os.path.join(out_dir, f'accuracy_{dataset_key}_std.csv')
    time_path = os.path.join(out_dir, f'flash_time_{dataset_key}.csv')

    mean_cols, std_cols, flash_times = {}, {}, {}
    if os.path.isfile(acc_path) and os.path.isfile(std_path):
        try:
            df_mean = pd.read_csv(acc_path, index_col='lambda')
            df_std = pd.read_csv(std_path, index_col='lambda')
            if (len(df_mean.index) == len(percentages) and
                    np.allclose(df_mean.index.values, percentages) and
                    len(df_std.index) == len(percentages) and
                    np.allclose(df_std.index.values, percentages)):
                mean_cols = {c: df_mean[c].tolist() for c in df_mean.columns}
                std_cols = {c: df_std[c].tolist() for c in df_std.columns}
        except Exception as ex:
            print(f'  [warn] could not parse existing results at {acc_path}: {ex}')
    if os.path.isfile(time_path):
        try:
            time_df = pd.read_csv(time_path)
            flash_times = dict(zip(time_df['variant'], time_df['flash_seconds']))
        except Exception as ex:
            print(f'  [warn] could not parse existing flash-time file at {time_path}: {ex}')
    return mean_cols, std_cols, flash_times


def run_dataset(dataset_key, variants, chunk_size, k, eval_epochs, eval_bs, eval_lr,
                itter, seed, test_size, force=False):
    cfg = DATASETS[dataset_key]
    if not os.path.isfile(cfg['source_csv']):
        print(f'[skip] {dataset_key}: source CSV not found -> {cfg["source_csv"]}')
        return None, None, None

    print(f'\n==================== {dataset_key} ({cfg["out_name"]}) ====================')

    out_dir = os.path.join(RESULTS_DIR, cfg['out_name'])
    mean_cols, std_cols, flash_times = (
        _load_existing_results(out_dir, dataset_key, PERCENTAGES) if not force else ({}, {}, {}))

    pending = {name: spec for name, spec in variants.items()
               if force or name not in mean_cols or name not in flash_times}
    already_done = [name for name in variants if name not in pending]
    if already_done:
        print(f'  [resume] {len(already_done)} variant(s) already have saved results, '
              f'skipping (pass --force to recompute): {already_done}')

    if pending:
        train_path, test_path, train_dir, n_train, n_test = make_holdout_split(
            dataset_key, cfg['source_csv'], seed, test_size)
        X_test, y_test = _read_chunk_xy(test_path, LABEL_CLASS)
        print(f'  fixed test set loaded ONCE: {len(X_test)} rows -- reused unchanged for every '
              f'variant, lambda, and retrain repeat; never pruned or resampled.')

        for variant_name in pending:
            split_by, use_hilbert = pending[variant_name]
            try:
                means, stds, flash_seconds = run_variant(
                    dataset_key, cfg, variant_name, split_by, use_hilbert, train_dir,
                    chunk_size, k, X_test, y_test, eval_epochs, eval_bs, eval_lr, itter, n_train)
                mean_cols[variant_name] = means
                std_cols[variant_name] = stds
                flash_times[variant_name] = flash_seconds
            except Exception as ex:
                print(f'  [error] variant {variant_name} failed: {ex}')
                traceback.print_exc()
    else:
        print(f'  [resume] all requested variants already completed for {dataset_key}; '
              f'nothing to recompute (pass --force to recompute).')

    if not mean_cols:
        print(f'[skip] {dataset_key}: no variant produced results')
        return None, None, None

    df_mean = pd.DataFrame(mean_cols, index=PERCENTAGES)
    df_std = pd.DataFrame(std_cols, index=PERCENTAGES)
    df_mean.index.name = 'lambda'
    df_std.index.name = 'lambda'

    out_dir = os.path.join(RESULTS_DIR, cfg['out_name'])
    os.makedirs(out_dir, exist_ok=True)

    acc_path = os.path.join(out_dir, f'accuracy_{dataset_key}.csv')
    acc_std_path = os.path.join(out_dir, f'accuracy_{dataset_key}_std.csv')
    df_mean.to_csv(acc_path, index=True)
    df_std.to_csv(acc_std_path, index=True)

    time_df = pd.DataFrame([{'variant': v, 'flash_seconds': s} for v, s in flash_times.items()])
    time_path = os.path.join(out_dir, f'flash_time_{dataset_key}.csv')
    time_df.to_csv(time_path, index=False)

    print(f'  saved -> {acc_path}')
    print(f'  saved -> {acc_std_path}')
    print(f'  saved -> {time_path}')
    return df_mean, df_std, time_df


def main():
    global STANDARDIZE, PREPROCESSING, R_MAX
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()))
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()))
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--k', type=int, default=K_SUBSETS)
    parser.add_argument('--eval-epochs', type=int, default=300)
    parser.add_argument('--eval-batch-size', type=int, default=512,
                        help='evaluation batch size (fixed at 512 for the VLDB submission)')
    parser.add_argument('--eval-lr', type=float, default=0.01)
    parser.add_argument('--itter', type=int, default=5,
                        help='retrain repeats per lambda (test set is fixed; only training '
                             'stochasticity varies across repeats)')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--no-standardize', action='store_true')
    parser.add_argument('--r-max', type=int, default=R_MAX,
                        help='FLA-SH round budget R_max (paper: 2; previously 1 was used implicitly)')
    parser.add_argument('--preprocessing', choices=PREPROCESSING_MODES, default=PREPROCESSING,
                        help="model-input preprocessing for valuation and retraining; 'legacy' = old behaviour")
    parser.add_argument('--quick', action='store_true',
                        help='tiny smoke run: few chunks/epochs/reps for a fast sanity check')
    parser.add_argument('--force', action='store_true',
                        help='recompute (dataset, variant) combos even if saved results '
                             'already exist in Results_selected/VLDB_submission/ '
                             '(default: skip already-executed combos)')
    args = parser.parse_args()

    STANDARDIZE = not args.no_standardize
    PREPROCESSING = args.preprocessing
    R_MAX = args.r_max

    if args.quick:
        args.chunk_size = 1000
        args.k = 5
        args.eval_epochs = 5
        args.itter = 2
        print('[quick] smoke-run settings: chunk_size=1000, k=5, eval_epochs=5, itter=2')

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass

    os.makedirs(RESULTS_DIR, exist_ok=True)
    long_acc_rows, long_time_rows = [], []

    for dataset_key in args.datasets:
        df_mean, df_std, time_df = run_dataset(
            dataset_key, VARIANTS if args.variants == list(VARIANTS.keys())
            else {k: VARIANTS[k] for k in args.variants},
            args.chunk_size, args.k, args.eval_epochs, args.eval_batch_size,
            args.eval_lr, args.itter, args.seed, args.test_size, force=args.force)
        if df_mean is None:
            continue
        for variant_name in df_mean.columns:
            for lam in df_mean.index:
                long_acc_rows.append({
                    'dataset': dataset_key, 'variant': variant_name, 'lambda': float(lam),
                    'accuracy_mean': df_mean.loc[lam, variant_name],
                    'accuracy_std': df_std.loc[lam, variant_name],
                })
        for _, row in time_df.iterrows():
            long_time_rows.append({'dataset': dataset_key, 'variant': row['variant'],
                                   'flash_seconds': row['flash_seconds']})

    if long_acc_rows:
        acc_all_path = os.path.join(RESULTS_DIR, 'vldb_tabular_accuracy_all.csv')
        pd.DataFrame(long_acc_rows).to_csv(acc_all_path, index=False)
        print(f'\nCombined accuracy results saved -> {acc_all_path}')
    if long_time_rows:
        time_all_path = os.path.join(RESULTS_DIR, 'vldb_tabular_flash_time_all.csv')
        pd.DataFrame(long_time_rows).to_csv(time_all_path, index=False)
        print(f'Combined FLA-SH timing results saved -> {time_all_path}')
    if not long_acc_rows:
        print('\nNo results produced.')


if __name__ == '__main__':
    main()
