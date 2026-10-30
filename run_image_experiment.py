import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import time
import random
import argparse
import traceback

import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.model_selection import train_test_split

from main import Data_Shapley_Dataset
from utils_dataShapley import create_model
from utils import get_metric

# Paths on the cluster; override with DATA_SHAPLEY_ROOT / IMAGE_DATA_ROOT.
PROJECT_ROOT = os.environ.get('DATA_SHAPLEY_ROOT', '/FLA-SH/')
IMAGE_DATA_ROOT = os.environ.get(
    'IMAGE_DATA_ROOT', '/data/Image_Data/')

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results')
WORK_ROOT = os.path.join(PROJECT_ROOT, 'Output', 'results', 'image')

MODEL_TYPE = 'AlexNet'
HIDDEN_LAYERS = [1024, 1024]
NUM_CLASSES = 10
LABEL_CLASS = 'class'

PERCENTAGES = np.arange(0.1, 0.35, 0.1, dtype=np.float64)

CHUNK_SIZE = 256
K_SUBSETS = 50
TH = 0.05


_IMAGE_CACHE = {}


def remap_old_root(path, old_prefixes=OLD_PATH_PREFIXES, new_root=NEW_PATH_ROOT):
    for old_prefix in old_prefixes:
        if path.startswith(old_prefix):
            return new_root + path[len(old_prefix):]
    return path


def update_img_fullpath(df, filepath, old_prefixes=OLD_PATH_PREFIXES):
    head_path, _ = os.path.split(filepath)
    for index, row in df.iterrows():
        fp = row['file_path']
        remapped = remap_old_root(fp, old_prefixes=old_prefixes)
        if remapped != fp:
            df.at[index, 'file_path'] = os.path.normpath(remapped)
        else:
            df.at[index, 'file_path'] = os.path.normpath(os.path.join(head_path, fp))
    return df


def load_image_cached(filename):
    filename = remap_old_root(filename)
    cached = _IMAGE_CACHE.get(filename)
    if cached is not None:
        return cached
    img = cv2.imread(filename)                   # BGR (H, W, C)
    if img is None:
        raise FileNotFoundError(
            f"cv2.imread failed for '{filename}' (after old-root remap). "
            f"Check IMAGE_DATA_ROOT / the file exists.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # -> RGB
    img = np.transpose(img)                      # -> (C, W, H)
    _IMAGE_CACHE[filename] = img
    return img


def load_images_from_filenames(filenames):
    return np.asarray([load_image_cached(fn) for fn in filenames])


def load_dataset_csv(csv_path, filepath_root, filename_col='file_path', label_col='class'):
    df = pd.read_csv(csv_path)
    df = update_img_fullpath(df=df, filepath=filepath_root)
    filenames = df[filename_col].to_list()
    labels = df[label_col].to_numpy()
    images_np = load_images_from_filenames(filenames)
    return images_np, labels, filenames


def predict_in_batches(model, X, batch_size=128):
    preds = []
    n = len(X)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            y_pred_batch = model.predict(X=X[start:end])
            if torch.is_tensor(y_pred_batch):
                y_pred_batch = y_pred_batch.detach().cpu()
            preds.append(y_pred_batch)
    if torch.is_tensor(preds[0]):
        return torch.cat(preds, dim=0)
    return np.concatenate(preds, axis=0)

VARIANTS = {
    'FLA-SH-KM-SR':   ('cluster', False),
    'FLA-SH-KDT-SR':  ('tree',    False),
    'FLA-SH-KM-HIL':  ('cluster', True),
    'FLA-SH-KDT-HIL': ('tree',    True),
    'FLA-SH-SR':      ('random',  False),
    'FLA-SH-HIL':     ('random',  True),
}

LABEL_TO_IDX_CIFAR = {
    'frog': 0, 'truck': 1, 'deer': 2, 'automobile': 3, 'bird': 4,
    'horse': 5, 'ship': 6, 'cat': 7, 'dog': 8, 'airplane': 9,
}
IDX_TO_LABEL_CIFAR = {v: k for k, v in LABEL_TO_IDX_CIFAR.items()}

_IDX_TO_LABEL_MNIST = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
}
_LABEL_TO_IDX_MNIST = {v: k for k, v in _IDX_TO_LABEL_MNIST.items()}

_IDX_TO_LABEL_FMNIST = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot',
}
_LABEL_TO_IDX_FMNIST = {v: k for k, v in _IDX_TO_LABEL_FMNIST.items()}

DATASETS = {
    'cifar': {
        'out_name': 'CIFAR-10',
        'source_csv': os.path.join(IMAGE_DATA_ROOT, 'CIFAR10', 'cifar_noise.csv'),
        'test_csv': os.path.join(IMAGE_DATA_ROOT, 'CIFAR10', 'cifar_test.csv'),
        'test_filepath_root': os.path.join(IMAGE_DATA_ROOT) + '/',
        'idx_to_label': IDX_TO_LABEL_CIFAR,
        'label_to_idx': LABEL_TO_IDX_CIFAR,
    },
    'mnist': {
        'out_name': 'MNIST',
        'source_csv': remap_old_root('/data/Image_Data/MNIST/train_noise.csv'),
        'test_csv': os.path.join(IMAGE_DATA_ROOT, 'MNIST', 'mnist_test.csv'),
        'test_filepath_root': os.path.join(IMAGE_DATA_ROOT) + '/',
        'idx_to_label': _IDX_TO_LABEL_MNIST,
        'label_to_idx': _LABEL_TO_IDX_MNIST,
    },
    'fmnist': {
        'out_name': 'F-MNIST',
        'source_csv': os.path.join(IMAGE_DATA_ROOT, 'f-MNIST', 'f-mnist_train_flipped.csv'),
        'test_csv': os.path.join(IMAGE_DATA_ROOT, 'f-MNIST', 'f-mnist_test.csv'),
        'test_filepath_root': os.path.join(IMAGE_DATA_ROOT) + '/',
        'idx_to_label': _IDX_TO_LABEL_FMNIST,
        'label_to_idx': _LABEL_TO_IDX_FMNIST,
    },
}


def load_fixed_test_set(test_csv, test_filepath_root, label_to_idx):
    """Load a dataset's test CSV and return (X, y) with integer labels."""
    X, labels, _ = load_dataset_csv(test_csv, test_filepath_root, label_col=LABEL_CLASS)
    y = np.array([label_to_idx.get(l, l) for l in labels]).astype(np.int64)
    return X, y

def evaluate_data_selection_fixed_test(scores_dict, ds_obj, X_test, y_test, percentages,
                                       eval_epochs, fit_batch_size, eval_lr, itter,
                                       predict_batch_size, label_to_idx, num_classes=NUM_CLASSES,
                                       early_stopping=0):
    scores_sorted = {name: val for name, val in
                     sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)}
    chunk_names = list(scores_sorted.keys())
    n_chunks = len(chunk_names)

    mean_lst, std_lst = [], []
    for perc in percentages:
        n_remove = int(np.ceil(n_chunks * perc))
        n_selected = max(1, n_chunks - n_remove)
        selected = chunk_names[:n_selected]

        filenames_lst = [fpath for name, fpath in ds_obj.datasets_filenames.items()
                         if name in selected]
        X_, y_ = [], []
        for fpath in filenames_lst:
            df = pd.read_csv(fpath)
            for _, row in df.iterrows():
                X_.append(load_image_cached(row['file_path']))
                y_.append(label_to_idx.get(row[LABEL_CLASS], row[LABEL_CLASS]))

        if not X_:
            print(f'[warn] no images resolved for lambda={perc:.2f}; skipping')
            mean_lst.append(np.nan)
            std_lst.append(np.nan)
            continue

        X_train = np.asarray(X_)
        y_train = np.asarray(y_).astype(np.int64)

        runs = []
        for _ in range(itter):
            X_tr2, X_val, y_tr2, y_val = train_test_split(
                X_train, y_train, test_size=0.1, train_size=0.9)

            model = create_model(ds_obj.model_type, problem=ds_obj.problem,
                                   max_epochs=eval_epochs, batch_size=fit_batch_size,
                                   lr=eval_lr, val_frac=0.1,
                                   hidden_layer_size=ds_obj.hidden_layers,
                                   use_img=True, use_NN=False, num_classes=num_classes,
                                   early_stopping=early_stopping)
            model.fit(X=X_tr2, y=y_tr2.ravel(), X_val=X_val, y_val=y_val,
                     batch_size=fit_batch_size, show_progr=True)

            y_pred = predict_in_batches(model, X_test, batch_size=predict_batch_size)
            y_pred = torch.argmax(y_pred, axis=1).cpu().detach().numpy()

            inv_map = getattr(model, '_label_map_inv', None)
            if inv_map:
                y_pred = np.array([inv_map.get(int(p), int(p)) for p in y_pred])

            runs.append(get_metric(metric='accuracy', y_pred=y_pred, y=y_test.ravel()))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        runs = np.asarray(runs, dtype=np.float64)
        mean_lst.append(float(runs.mean()))
        std_lst.append(float(runs.std(ddof=1)) if len(runs) > 1 else 0.0)
        print(f'    lambda={perc:.2f}  selected {n_selected}/{n_chunks} train chunks  '
              f'acc(on FIXED test)={runs.mean():.4f} +/- {std_lst[-1]:.4f}  runs={np.round(runs, 4)}')

    return mean_lst, std_lst


def run_variant(dataset_key, cfg, variant_name, split_by, use_hilbert, train_csv,
                chunk_size, k, X_test, y_test, eval_epochs, fit_batch_size, eval_lr,
                itter, predict_batch_size, n_train, r_max, epsilon, early_stopping=0):
    out_dir = os.path.join(WORK_ROOT, dataset_key, variant_name)
    ds = Data_Shapley_Dataset(
        datasets_path=train_csv,
        outpath=out_dir,
        metric='Accuracy',
        problem='classification',
        model_type=MODEL_TYPE,
        num_of_classes=NUM_CLASSES,
        hidden_layers=list(HIDDEN_LAYERS),
    )

    print(f'  [{variant_name}] computing FLA-SH values on the WHOLE (unsplit) train csv '
          f'(split_by={split_by}, hilbert={use_hilbert}, chunk_size={chunk_size}, k={k}, '
          f'r_max={r_max}, epsilon={epsilon}) ...')
    t0 = time.perf_counter()
    out_shapley_values, _ = ds.execute_per_dataset(
        split_data=True, th=TH, chunk_size=chunk_size, split_by=split_by,
        data_type='img', k=k, smart_selection=True, use_hilbert_kernel=use_hilbert,
        use_perm=True, idx_to_label=cfg['idx_to_label'], convert_label=True,
        r_max=r_max, err=epsilon)
    flash_seconds = time.perf_counter() - t0
    inner = out_shapley_values[next(iter(out_shapley_values))]

    total_rows = sum(len(pd.read_csv(fp)) for fp in ds.datasets_filenames.values())
    assert total_rows == n_train, (
        f"BUG: chunks cover {total_rows} rows, expected exactly n_train={n_train} "
        f"(the whole train csv).")

    print(f'  [{variant_name}] {len(inner)} chunks scored '
          f'({total_rows} rows, verified == whole train csv n_train={n_train}) in '
          f'{flash_seconds:.2f}s; evaluating against the fixed, pre-made test set ...')

    means, stds = evaluate_data_selection_fixed_test(
        scores_dict=inner, ds_obj=ds, X_test=X_test, y_test=y_test,
        percentages=PERCENTAGES, eval_epochs=eval_epochs, fit_batch_size=fit_batch_size,
        eval_lr=eval_lr, itter=itter, predict_batch_size=predict_batch_size,
        label_to_idx=cfg['label_to_idx'], early_stopping=early_stopping)
    return means, stds, flash_seconds


def _load_existing_results(out_dir, dataset_key, percentages):
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


def run_dataset(dataset_key, variants, chunk_size, k, eval_epochs, fit_batch_size,
                eval_lr, itter, predict_batch_size, seed, force=False, r_max=1, epsilon=0.001,
                early_stopping=0):
    cfg = DATASETS[dataset_key]
    if not os.path.isfile(cfg['source_csv']):
        print(f'[skip] {dataset_key}: source CSV not found -> {cfg["source_csv"]}\n'
              f'  (image data lives on the remote machine; set IMAGE_DATA_ROOT / '
              f'IMAGE_DATA_NEW_ROOT env vars if running there.)')
        return None, None, None
    if not cfg.get('test_csv'):
        print(f'[skip] {dataset_key}: no pre-made test csv configured '
              f'(DATASETS["{dataset_key}"]["test_csv"] is None) -- see the module '
              f'docstring; set it to a real path once one is confirmed.')
        return None, None, None
    if not os.path.isfile(cfg['test_csv']):
        print(f'[skip] {dataset_key}: configured test CSV not found -> {cfg["test_csv"]}')
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
        train_path = cfg['source_csv']
        n_train = len(pd.read_csv(train_path))
        X_test, y_test = load_fixed_test_set(cfg['test_csv'], cfg['test_filepath_root'], cfg['label_to_idx'])
        print(f'  train csv used WHOLE, unsplit: {n_train} images -> {train_path}')
        print(f'  fixed test set loaded ONCE from {cfg["test_csv"]}: {len(X_test)} images -- '
              f'reused unchanged for every variant, lambda, and retrain repeat; never pruned or resampled.')

        for variant_name in pending:
            split_by, use_hilbert = pending[variant_name]
            try:
                means, stds, flash_seconds = run_variant(
                    dataset_key, cfg, variant_name, split_by, use_hilbert, train_path,
                    chunk_size, k, X_test, y_test, eval_epochs, fit_batch_size, eval_lr,
                    itter, predict_batch_size, n_train, r_max, epsilon,
                    early_stopping=early_stopping)
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()))
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()))
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--k', type=int, default=K_SUBSETS)
    parser.add_argument('--eval-epochs', type=int, default=50)
    parser.add_argument('--eval-batch-size', type=int, default=512,
                        help='batch size for retraining and prediction (paper: 512)')
    parser.add_argument('--eval-lr', type=float, default=0.05)
    parser.add_argument('--itter', type=int, default=5)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='early-stopping patience in epochs for retraining (0 = off)')
    parser.add_argument('--r-max', type=int, default=1,
                        help='number of FLA-SH rounds R_max (the paper uses 2)')
    parser.add_argument('--epsilon', type=float, default=0.001,
                        help='Eq. 4 stopping tolerance, checked from round 2 on')
    parser.add_argument('--quick', action='store_true',
                        help='fast smoke run with few chunks, epochs and repeats')
    parser.add_argument('--force', action='store_true',
                        help='recompute variants that already have saved results')
    args = parser.parse_args()

    if args.quick:
        args.chunk_size = 1000
        args.k = 5
        args.eval_epochs = 3
        args.itter = 2
        print('[quick] smoke-run settings: chunk_size=1000, k=5, eval_epochs=3, itter=2')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    long_acc_rows, long_time_rows = [], []

    variants = VARIANTS if args.variants == list(VARIANTS.keys()) else {
        k_: VARIANTS[k_] for k_ in args.variants}

    for dataset_key in args.datasets:
        df_mean, df_std, time_df = run_dataset(
            dataset_key, variants, args.chunk_size, args.k, args.eval_epochs,
            args.eval_batch_size, args.eval_lr, args.itter, args.eval_batch_size,
            args.seed, force=args.force, r_max=args.r_max, epsilon=args.epsilon,
            early_stopping=args.early_stopping)
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
        acc_all_path = os.path.join(RESULTS_DIR, 'vldb_image_accuracy_all.csv')
        pd.DataFrame(long_acc_rows).to_csv(acc_all_path, index=False)
        print(f'\nCombined accuracy results saved -> {acc_all_path}')
    if long_time_rows:
        time_all_path = os.path.join(RESULTS_DIR, 'vldb_image_flash_time_all.csv')
        pd.DataFrame(long_time_rows).to_csv(time_all_path, index=False)
        print(f'Combined FLA-SH timing results saved -> {time_all_path}')
    if not long_acc_rows:
        print('\nNo results produced.')


if __name__ == '__main__':
    main()
