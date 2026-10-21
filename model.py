import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import minmax_scale
from collections import Counter

from utils import transform_to_torch, get_metric, GroupDRO, group_dro_loss, cvar_loss, variance_group_loss

def _flat_labels(y):
  
    if isinstance(y, list):
        if all(isinstance(i, pd.DataFrame) for i in y):
            return pd.concat(y, axis=0).to_numpy().ravel()
        elif all(isinstance(i, pd.Series) for i in y):
            return pd.concat(y).to_numpy().ravel()
        elif all(isinstance(i, np.ndarray) for i in y):
            return np.concatenate(y, axis=0).ravel()
        else:
            parts = []
            for i in y:
                if hasattr(i, 'to_numpy'):
                    parts.append(i.to_numpy().ravel())
                else:
                    parts.append(np.asarray(i).ravel())
            return np.concatenate(parts)
    elif isinstance(y, (pd.DataFrame, pd.Series)):
        return y.to_numpy().ravel()
    else:
        return np.asarray(y).ravel()


def _min_split_n(val_fraction):
    return max(2, int(np.ceil(1.0 / val_fraction)) + 1)


class NN_model(nn.Module):

    IMG_MICRO_BATCH = 128

    BUILD_NEEDS_ALL_X = False

    def __init__(self, mode_problem='classification', use_img=False,
                 hidden_layers=[100], optimizer='sgd', batch_size=128,
                 max_epoch=100, validation_fraction=0.1, early_stopping=5,
                 address=None, test_batch_size=1000, random_seed=666,
                 lr=0.001, num_of_classes=None, normalize=True, device=None,
                 use_fairness=False, fairness_attr=0, num_of_fair=2,
                 lambda_fair=1.0, fair_metric='groupdro', compute_Shapley=True,
                 early_stop_min_delta=0.001):
        super().__init__()
        self.problem  = mode_problem
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = max_epoch
        self.val_fraction = validation_fraction
        self.early_stop = early_stopping
        self.early_stop_min_delta = early_stop_min_delta
        self.address = address
        self._extra_train_ops = []
        self.random_seed = random_seed
        self.normalize = normalize
        self.use_img = use_img
        self.num_classes = num_of_classes
        self.compute_Shapley = compute_Shapley
 
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
 
        if self.problem.lower() not in ('classification', 'regression'):
            raise ValueError('Problem must be either classification or regression')
        self.is_built = False
 
        self.use_fairness = use_fairness
        self.fairness_attr = fairness_attr
        self.num_of_fair = num_of_fair
        self.lambda_fair = lambda_fair
        self.fair_metric = fair_metric
 
        self._label_map     = None   
        self._label_map_inv = None  
 
    
    def _small_chunk_val_split(self, X, y, fair_attrb=None):
        
        n = X.shape[0] if isinstance(X, np.ndarray) else len(X)
 
        if n <= 1:
            return X, X, y, y, fair_attrb, fair_attrb
 
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[:-1], X.iloc[[-1]]
        else:
            X_train, X_val = X[:-1], X[[-1]]
 
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_train, y_val = y.iloc[:-1], y.iloc[[-1]]
        else:
            y_train, y_val = y[:-1], y[[-1]]
 
        if fair_attrb is None:
            return X_train, X_val, y_train, y_val, None, None
 
        if isinstance(fair_attrb, (pd.DataFrame, pd.Series)):
            f_train = fair_attrb.iloc[:-1]
            f_val = fair_attrb.iloc[[-1]]
        else:
            f_train = fair_attrb[:-1]
            f_val = fair_attrb[[-1]]
 
        return X_train, X_val, y_train, y_val, f_train, f_val
 
    
    def _build_label_map(self, y_flat):
        unique_sorted = sorted(set(int(v) for v in y_flat))
        if unique_sorted == list(range(len(unique_sorted))):
            self._label_map = None
            self._label_map_inv = None
        else:
            self._label_map = {orig: new for new, orig in enumerate(unique_sorted)}
            self._label_map_inv = {new: orig for orig, new in self._label_map.items()}
 
    def _remap_labels(self, y):
        if self._label_map is None:
            return y
 
        lm = self._label_map
 
        if isinstance(y, pd.DataFrame):
            _mapper = getattr(y, 'map', None)
            if _mapper is not None and callable(_mapper):
                import pandas as _pd
                _version = tuple(int(x) for x in _pd.__version__.split('.')[:2])
                if _version >= (2, 1):
                    return y.map(lambda v: lm.get(int(v), int(v)))
            return y.applymap(lambda v: lm.get(int(v), int(v)))
        if isinstance(y, pd.Series):
            return y.map(lambda v: lm.get(int(v), int(v)))
        if isinstance(y, np.ndarray):
            return np.vectorize(lambda v: lm.get(int(v), int(v)))(y)
        if isinstance(y, list):
            return [lm.get(int(v), int(v)) for v in y]
        return y
 
    
    def fit(self, X, y, X_val, y_val, use_val=True, epochs=None,
            batch_size=None, save=False, load=False, sample_weight=None,
            data_dict=None, use_chunks=True, metric='accuracy',
            show_progr=False, fair_attrbs=None, fair_attrbs_val=None,
            release_inputs=False):

        all_y = _flat_labels(y)
        if self.num_classes is None:
            self.num_classes = max(2, len(np.unique(all_y)))
 
        if self.problem == 'classification':
            self._build_label_map(all_y)
            unique_labels = sorted(set(int(v) for v in all_y))
            required_classes = max(unique_labels) + 1   # must cover max label
            if self.num_classes < required_classes:
                self.num_classes = required_classes

            y = self._remap_y_structure(y)
            if y_val is not None and not isinstance(y_val, bool):
                y_val = self._remap_y_structure(y_val)
 
        self.metric = metric
 
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
 
        if not self.is_built:
            if isinstance(X, list) and X and not self.BUILD_NEEDS_ALL_X:
                # Shape is all build() needs -- see BUILD_NEEDS_ALL_X.
                first = X[0]
                all_X = first.to_numpy() if hasattr(first, 'to_numpy') else np.asarray(first)
            elif isinstance(X, list):
                if all(isinstance(i, pd.DataFrame) for i in X):
                    all_X = pd.concat(X, axis=0).to_numpy()
                elif all(isinstance(i, np.ndarray) for i in X):
                    all_X = np.concatenate(X, axis=0)
                else:
                    all_X = np.concatenate([
                        i.to_numpy() if hasattr(i, 'to_numpy') else np.asarray(i)
                        for i in X], axis=0)
            elif isinstance(X, pd.DataFrame):
                all_X = X.to_numpy()
            else:
                all_X = np.asarray(X)
 
            self.build(all_X, all_y)
            del all_X   # otherwise held for the whole training run

        fair_train = None
        fair_val = None
        owns_split = False   # True once X_train_lst/X_val_lst are lists fit() built itself
 
        if X is None or len(X) == 0:
            return
 
        min_split = _min_split_n(self.val_fraction)
 
        if use_chunks:
            if X_val is None and use_val:
                X_train_lst = []
                X_val_lst = []
                y_train_lst = []
                y_val_lst = []
                fair_train = []
                fair_val   = []
                owns_split = True

                if isinstance(X, (pd.DataFrame, np.ndarray)) and \
                   isinstance(y, (pd.DataFrame, np.ndarray)):
                    n = X.shape[0] if isinstance(X, np.ndarray) else len(X)
                    fa = fair_attrbs if self.use_fairness else None
                    if n < min_split:
                        Xt, Xv, yt, yv, ft, fv = self._small_chunk_val_split(X, y, fa)
                    else:
                        if self.use_fairness and fa is not None:
                            Xt, Xv, yt, yv, ft, fv = train_test_split(X, y, fa, test_size=self.val_fraction)
                        else:
                            Xt, Xv, yt, yv = train_test_split(X, y, test_size=self.val_fraction)
                            ft = None
                            fv = None
                    X_train_lst.append(Xt)
                    X_val_lst.append(Xv)
                    y_train_lst.append(yt)
                    y_val_lst.append(yv)
                    if self.use_fairness:
                        fair_train.append(ft); fair_val.append(fv)

                elif isinstance(X, list) and isinstance(y, list):
                    fa_list = (fair_attrbs if (self.use_fairness and isinstance(fair_attrbs, list)) else [None] * len(X))
                    for i_sub, (X_, y_, fa_) in enumerate(zip(X, y, fa_list)):
                        n = X_.shape[0] if isinstance(X_, np.ndarray) else len(X_)
                        if n < min_split:
                            Xt, Xv, yt, yv, ft, fv = self._small_chunk_val_split( X_, y_, fa_)
                        else:
                            if self.use_fairness and fa_ is not None:
                                Xt, Xv, yt, yv, ft, fv = train_test_split(X_, y_, fa_, test_size=self.val_fraction)
                            else:
                                Xt, Xv, yt, yv = train_test_split(X_, y_, test_size=self.val_fraction)
                                ft = None
                                fv = None
                        X_train_lst.append(Xt)
                        X_val_lst.append(Xv)
                        y_train_lst.append(yt)
                        y_val_lst.append(yv)
                        if self.use_fairness:
                            fair_train.append(ft)
                            fair_val.append(fv)
                        if release_inputs:
                            X[i_sub] = None
            else:
                if self.use_fairness:
                    fair_train = fair_attrbs
                    fair_val = fair_attrbs_val
                X_train_lst = X
                y_train_lst = y
                X_val_lst = X_val
                y_val_lst = y_val
        else:
            # use_chunks=False path
            if X_val is None and self.val_fraction * len(X) > 2 and use_val:
                if self.use_fairness and fair_attrbs is not None:
                    X_train_lst, X_val_lst, y_train_lst, y_val_lst, fair_train, fair_val = train_test_split(X, y, fair_attrbs, test_size=self.val_fraction)
                else:
                    X_train_lst, X_val_lst, y_train_lst, y_val_lst = train_test_split(X, y, test_size=self.val_fraction)
            else:
                if isinstance(X, pd.DataFrame): X = X.to_numpy()
                if isinstance(y, pd.DataFrame): y = y.to_numpy()
                if isinstance(X_val, pd.DataFrame): X_val = X_val.to_numpy()
                if isinstance(y_val, pd.DataFrame): y_val = y_val.to_numpy()
                X_train_lst = X
                y_train_lst = y
                X_val_lst = X_val
                y_val_lst = y_val
                if self.use_fairness:
                    fair_train = fair_attrbs
                    fair_val = fair_attrbs_val
 
        self.training_(X=X_train_lst, y=y_train_lst,
                       X_val=X_val_lst, y_val=y_val_lst,
                       fair_train=fair_train, fair_val=fair_val,
                       use_val=use_val, max_epochs=epochs, batch_size=batch_size,
                       data_dict=data_dict, show_progr=show_progr,
                       normalize_inplace=owns_split)

    def _remap_y_structure(self, y):
        if self._label_map is None:
            return y
        if isinstance(y, list):
            return [self._remap_labels(yi) for yi in y]
        return self._remap_labels(y)
 
    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True
        if self.num_classes is None:
            self.num_classes = max(2, len(np.unique(_flat_labels(y))))
        else:
            self.num_classes = max(2, self.num_classes)
 
        self.input_dim = X.shape[1]
        self.layers_ = nn.Sequential()
 
        hidden_dim = self.hidden_layers[0]
        input_layer = nn.Linear(self.input_dim, hidden_dim, bias=bias, device=self.device, dtype=torch.float)
        if init_:
            nn.init.kaiming_uniform_(input_layer.weight, a=np.sqrt(5))
        self.layers_.add_module('linear_0', input_layer)
        self.layers_.add_module('activation_0', nn.ReLU())
 
        for i, layer_dim in enumerate(self.hidden_layers):
            linear = nn.Linear(hidden_dim, layer_dim, bias=bias, device=self.device, dtype=torch.float)
            hidden_dim = layer_dim
            if init_:
                nn.init.kaiming_uniform_(linear.weight, a=np.sqrt(5))
            self.layers_.add_module(f'linear_{i + 1}', linear)
            self.layers_.add_module(f'activation_{i + 1}', nn.ReLU())
 
        out_layer = nn.Linear(hidden_dim, 1 if self.problem == 'regression' else self.num_classes, bias=bias, dtype=torch.float, device=self.device)
        if init_:
            nn.init.kaiming_uniform_(out_layer.weight, a=np.sqrt(5))
        self.layers_.add_module('out_layer', out_layer)
 
 
    def training_(self, X, y, X_val, y_val, fair_train, fair_val,
                  use_val, max_epochs, batch_size, data_dict, show_progr,
                  normalize_inplace=False):
        assert len(X) == len(y), 'Input and labels not the same size'
 
        self.history = {'metrics': [], 'idxs': [], 'keys': []}
        stop_counter = 0
        higher_is_better = self.metric in ['accuracy', 'auc', 'f1']
        best_performance = -np.inf if higher_is_better else np.inf

        _momentum = getattr(self, 'momentum', 0.0)
        _weight_decay = getattr(self, 'weight_decay', 0.0)
        if self.optimizer.lower() == 'sgd':
            optimiser = torch.optim.SGD(self.parameters(), lr=self.lr,
                                        momentum=_momentum, weight_decay=_weight_decay)
        else:
            optimiser = torch.optim.Adam(self.parameters(), lr=self.lr,
                                         weight_decay=_weight_decay)

        scheduler = None
        if getattr(self, 'lr_schedule', None) == 'cosine' and max_epochs > 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=max_epochs)

        def _normalize_list(lst):
            if normalize_inplace:
                for i, a in enumerate(lst):
                    lst[i] = preprocessing.normalize(a) if a is not None else None
                return lst
            return [preprocessing.normalize(a) if a is not None else None for a in lst]

        if self.normalize:
            if isinstance(X, list):
                if not self.use_img:
                    X = _normalize_list(X)
                    if X_val is not None:
                        if isinstance(X_val, list):
                            X_val = _normalize_list(X_val)
                        elif isinstance(X_val, np.ndarray):
                            X_val = preprocessing.normalize(X_val)
            elif isinstance(X, np.ndarray):
                if not self.use_img:
                    X = preprocessing.normalize(X)
                    if X_val is not None:
                        if isinstance(X_val, list):
                            X_val = [preprocessing.normalize(a) if a is not None else None for a in X_val]
                        elif isinstance(X_val, np.ndarray):
                            X_val = preprocessing.normalize(X_val)

        loss_fair_fn = None
        if self.use_fairness and self.fair_metric == 'groupdro':
            loss_fair_fn = GroupDRO(n_groups=self.num_of_fair, device=self.device)

        for epoch in range(max_epochs):
            idxs, val_metrics, keys_lst = self.one_epoch_iter(
                X=X, y=y, X_val=X_val, y_val=y_val,
                use_val=use_val, fair_train=fair_train, fair_val=fair_val,
                batch_size=batch_size, data_dict=data_dict, optimiser=optimiser,
                loss_fair_fn=loss_fair_fn)
 
            self.history['metrics'] = val_metrics
            self.history['idxs'] = idxs
            self.history['keys'] = keys_lst

            if scheduler is not None:
                scheduler.step()

            current_performance = (np.mean(val_metrics) if (use_val and val_metrics) else float('nan'))
 
            if epoch % 10 == 0 and show_progr:
                print(f'Epoch {epoch}/{max_epochs}, '
                      f'avg_val_metric: {current_performance:.4f}, '
                      f'best: {best_performance:.4f}')

            if X_val is not None and val_metrics:
                _min_delta = getattr(self, 'early_stop_min_delta', 0.001)
                improved = (current_performance > best_performance + _min_delta
                            if higher_is_better
                            else current_performance < best_performance - _min_delta)
                if improved:
                    best_performance = current_performance
                    stop_counter = 0
                else:
                    stop_counter += 1

                if self.early_stop > 0 and stop_counter > self.early_stop:
                    print(f'Early stopping at epoch {epoch}. '
                          f'Best: {best_performance:.4f}')
                    break
 
    def one_epoch_iter(self, X, y, X_val, y_val, fair_train, fair_val,
                       use_val, batch_size, optimiser, data_dict=None, loss_fair_fn=None):
        val_metrics = []
        keys_lst = []
 
        if self.use_img:
            if data_dict is None:
                idxs = np.random.permutation(len(X))
                batches = self._make_img_batches(X, y, X_val, y_val, fair_train, fair_val, use_val, batch_size, idxs)
            else:
                idxs = np.random.permutation(len(data_dict.keys()))
                keys_lst = list(data_dict.keys())
                batches = [(keys_lst[int(i)], X[i], y[i], X_val[i], y_val[i], None, None) for i in idxs]
        else:
            if data_dict is None:
                idxs = np.random.permutation(len(X))
                batches = self._make_tabular_batches( X, y, X_val, y_val, fair_train, fair_val, use_val, batch_size, idxs)
            else:
                idxs = np.random.permutation(len(data_dict.keys()))
                keys_lst = list(data_dict.keys())
                if self.use_fairness:
                    batches = [(keys_lst[int(i)], X[i], y[i],
                                X_val[i], y_val[i],
                                fair_train[i], fair_val[i]) for i in idxs]
                else:
                    batches = [(keys_lst[int(i)], X[i], y[i],
                                X_val[i], y_val[i], None, None) for i in idxs]

        if self.problem == "classification":
            loss_fn = nn.CrossEntropyLoss(reduction='mean').to(self.device)
            loss_fn_none = nn.CrossEntropyLoss(reduction='none').to(self.device)
        else:
            loss_fn = nn.MSELoss(reduction='mean').to(self.device)
            loss_fn_none = nn.MSELoss(reduction='none').to(self.device)
 
        for batch in batches:
            (key_, X_batch, y_batch, X_val_batch, y_val_batch, fair_train_batch, fair_val_batch) = batch

            self.train()

            y_batch = transform_to_torch(y_batch).to(self.device).ravel()

            if self.use_fairness:
                fair_train_batch = transform_to_torch(fair_train_batch).to(self.device).long()
                if use_val and fair_val_batch is not None:
                    fair_val_batch = transform_to_torch(fair_val_batch).to(self.device).long()
                else:
                    fair_val_batch = None


            has_val = (use_val and X_val_batch is not None and y_val_batch is not None)

            if has_val:
                X_val_batch = transform_to_torch(X_val_batch).to(self.device)
                y_val_batch = transform_to_torch(y_val_batch).to(self.device).ravel()
                if X_val_batch.ndim == 1:
                    X_val_batch = X_val_batch.unsqueeze(0)

            optimiser.zero_grad()

            X_np = np.asarray(X_batch) if isinstance(X_batch, list) else X_batch
            if isinstance(X_np, np.ndarray) and X_np.ndim == 5 and X_np.shape[0] == 1:
                X_np = X_np[0]
            n_samples = X_np.shape[0] if hasattr(X_np, 'shape') else len(X_np)

            use_micro = (self.use_img and not self.use_fairness
                         and n_samples > self.IMG_MICRO_BATCH)

            per_sample_losses = None

            if use_micro:
                if self.problem == "classification":
                    y_batch = y_batch.long()
                else:
                    y_batch = y_batch.reshape(-1, 1)

                loss_value = 0.0
                for start in range(0, n_samples, self.IMG_MICRO_BATCH):
                    end = min(start + self.IMG_MICRO_BATCH, n_samples)
                    X_mb = transform_to_torch(X_np[start:end]).to(self.device)
                    y_mb = y_batch[start:end]
                    if X_mb.ndim == 3:
                        X_mb = X_mb.unsqueeze(0)

                    y_pred_mb = self.predict(X_mb)
                    loss_mb = loss_fn(y_pred_mb, y_mb)
                    (loss_mb * ((end - start) / n_samples)).backward()
                    loss_value += loss_mb.item() * (end - start) / n_samples

                    del X_mb, y_pred_mb, loss_mb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                X_batch = transform_to_torch(X_batch).to(self.device)
                if X_batch.ndim == 1:
                    X_batch = X_batch.unsqueeze(0)
                if self.use_img and X_batch.ndim == 3:
                    X_batch = X_batch.unsqueeze(0)

                y_pred = self.predict(X_batch)

                if self.problem == "classification":
                    y_batch = y_batch.long()
                    loss = loss_fn(y_pred, y_batch)
                    if self.use_fairness:
                        per_sample_losses = loss_fn_none(y_pred, y_batch)
                else:
                    y_batch = y_batch.reshape(-1, 1)
                    loss = loss_fn(y_pred, y_batch)
                    if self.use_fairness:
                        per_sample_losses = loss_fn_none(y_pred, y_batch)
                        if per_sample_losses.ndim > 1:
                            per_sample_losses = per_sample_losses.mean(dim=1)

                if self.use_fairness and per_sample_losses is not None:
                    if self.fair_metric == 'groupdro':
                        loss = loss + self.lambda_fair * group_dro_loss(loss_fair_fn, fair_train_batch, per_sample_losses)
                    elif self.fair_metric == 'cvar':
                        loss = loss + self.lambda_fair * cvar_loss(per_sample_losses=per_sample_losses)
                    elif self.fair_metric == 'varianceloss':
                        loss = loss + self.lambda_fair * variance_group_loss(groups=fair_train_batch, per_sample_losses=per_sample_losses, lambda_var=1)

                loss.backward()

            optimiser.step()
            keys_lst.append(key_)

            if has_val:
                val_metric = self.validation(X=X_val_batch, y=y_val_batch,
                                             fair_val=fair_val_batch,
                                             loss_fn=loss_fn_none,
                                             loss_fair_fn=loss_fair_fn if self.use_fairness else None)
                val_metrics.append(val_metric)

        return idxs, val_metrics, keys_lst
 
    def _make_tabular_batches(self, X, y, X_val, y_val,
                               fair_train, fair_val, use_val,
                               batch_size, idxs):
        def _xsl(arr, k):
            if isinstance(arr, pd.DataFrame): return arr.iloc[k]
            return arr[k]
 
        def _vitem(v, k):
            if v is None: return None
            if isinstance(v, list): return v[k] if k < len(v) else None
            return v   
 
        def _fitem(f, k):
            if f is None: return None
            if isinstance(k, slice):
                if isinstance(f, (pd.DataFrame, pd.Series)):
                    return f.iloc[k]
                if isinstance(f, list):
                    return f[k]
                return f[k]   
            if isinstance(f, list): return f[k] if k < len(f) else None
            if isinstance(f, (pd.DataFrame, pd.Series)):
                return f.iloc[k] if k < len(f) else None
            return f[k] if k < len(f) else None
 
        n_batches = int(np.ceil(len(idxs) / batch_size))
        batches = []
        for b in range(n_batches):
            ks = b * batch_size
            ke = (b + 1) * batch_size
            bi = idxs[ks:ke]

            if isinstance(X, list) and isinstance(y, list):
                k = int(idxs[ks])
                Xb = X[k]
                yb = y[k]
                Xvb = _vitem(X_val, k)
                yvb = _vitem(y_val, k)
                ftb = _fitem(fair_train, k)
                fvb = _fitem(fair_val,   k)
            else:
                sl = slice(ks, ke)
                Xb = _xsl(X, sl)
                yb = _xsl(y, sl)
                Xvb = _vitem(X_val, sl)
                yvb = _vitem(y_val, sl)
                ftb = _fitem(fair_train, sl)
                fvb = _fitem(fair_val, sl)

            batches.append((bi, Xb, yb, Xvb, yvb, ftb, fvb))
        return batches
 
    def _make_img_batches(self, X, y, X_val, y_val,
                           fair_train, fair_val, use_val,
                           batch_size, idxs):
        if not use_val:
            X_val, y_val, fair_val = None, None, None

        def _s(arr, sl):
            if arr is None: return None
            if isinstance(arr, (pd.DataFrame, pd.Series)): return arr.iloc[sl]
            if isinstance(arr, list): return arr[sl]
            return arr[sl]

        n_batches = int(np.ceil(len(idxs) / batch_size))
        batches = []
        for b in range(n_batches):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            bi = idxs[sl]
            Xv = X_val if isinstance(X_val, np.ndarray) else _s(X_val, sl)
            yv = y_val if isinstance(y_val, np.ndarray) else _s(y_val, sl)
            ft = _s(fair_train, sl) if fair_train is not None else None
            fv = _s(fair_val,   sl) if fair_val   is not None else None
            batches.append((bi, _s(X, sl), _s(y, sl), Xv, yv, ft, fv))
        return batches
 
    def validation(self, X, y, fair_val, loss_fn, loss_fair_fn):
        self.eval()
        with torch.no_grad():
            if self.use_img and X.shape[0] > self.IMG_MICRO_BATCH:
                preds = []
                for start in range(0, X.shape[0], self.IMG_MICRO_BATCH):
                    end = min(start + self.IMG_MICRO_BATCH, X.shape[0])
                    preds.append(self.predict(X[start:end]))
                y_pred = torch.cat(preds, dim=0)
            else:
                y_pred = self.predict(X)

            if self.problem == "classification":
                y = y.long()
                y_pred_metric = torch.argmax(y_pred, dim=1).cpu().numpy()
                val_loss_per_sample = loss_fn(y_pred, y)
            else:
                y = y.reshape(-1, 1)
                y_pred_metric = y_pred.cpu().numpy()
                val_loss_per_sample = loss_fn(y_pred, y)
                if val_loss_per_sample.ndim > 1:
                    val_loss_per_sample = val_loss_per_sample.mean(dim=1)
 
            y_true = y.cpu().numpy()
            metric_val = get_metric(metric=self.metric,
                                    y_pred=y_pred_metric, y=y_true,
                                    use_fairness=self.use_fairness)

            if self.use_fairness and fair_val is not None and self.compute_Shapley:
                n = val_loss_per_sample.shape[0]
                if len(fair_val) != n:
                    fair_val = fair_val[:n]

                if self.fair_metric == 'groupdro':
                    metric_val -= (self.lambda_fair * group_dro_loss(loss_fair_fn, fair_val, val_loss_per_sample)).item()
                elif self.fair_metric == 'cvar':
                    metric_val -= cvar_loss(per_sample_losses=val_loss_per_sample).item()
                elif self.fair_metric == 'varianceloss':
                    metric_val -= variance_group_loss(groups=fair_val, per_sample_losses=val_loss_per_sample, lambda_var=1).item()

            return metric_val
 
    def predict(self, X):
        return self.layers_(X)



class LSTM_Model(NN_model):

    BUILD_NEEDS_ALL_X = True   # vocab_size = max token id over ALL of X

    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True

        flat = _flat_labels(y)
        if self.num_classes is None and self.problem == "classification":
            self.num_classes = max(2, len(np.unique(flat)))
    
        if X is not None and X.size > 0:
            max_token_id = int(np.nanmax(X))
            vocab_size = max(2, max_token_id + 1)
        else:
            vocab_size = 20002

        embedding_dim = self.hidden_layers[0]  if self.hidden_layers else 64
        hidden_dim = self.hidden_layers[-1] if self.hidden_layers else 64

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0, device=self.device)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, device=self.device)
        self.out_layer = nn.Linear(hidden_dim, 1 if self.problem == 'regression' else self.num_classes, bias=bias, dtype=torch.float, device=self.device)

        if init_:
            nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
            nn.init.xavier_uniform_(self.out_layer.weight)

    def predict(self, X):
        if X.ndim == 1:
            X = X.unsqueeze(0)
        vocab_size = self.embedding.num_embeddings
        X = X.long().clamp(min=0, max=vocab_size - 1)
        embeds = self.embedding(X)
        _, (hidden, _) = self.lstm(embeds)
        return self.out_layer(hidden[-1])



class CNN_Model(NN_model):

    def __init__(self, mode_problem='classification', hidden_layers=[100],
                 optimizer='sgd', batch_size=128, max_epoch=100,
                 validation_fraction=0.1, early_stopping=0, address=None,
                 test_batch_size=1000, random_seed=666, lr=0.001,
                 normalize=True, device=None, strides=[2], kernel_sizes=[2],
                 channels=[1], global_averaging=False, dropout=0,
                 num_of_classes=2, use_fairness=False, fairness_attr=[0],
                 num_of_fair=2, lambda_fair=1.0, fair_metric='groupdro',
                 compute_Shapley=True):

        super().__init__(mode_problem, True, hidden_layers, optimizer,
                         batch_size, max_epoch, validation_fraction,
                         early_stopping, address, test_batch_size,
                         random_seed, lr, num_of_classes, normalize, device,
                         use_fairness, fairness_attr, num_of_fair,
                         lambda_fair, fair_metric, compute_Shapley)
        self.strides = ([1] * len(kernel_sizes) if strides is None else strides)
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        assert len(channels) == len(kernel_sizes), 'Invalid channels or kernel_sizes'
        self.global_averaging = global_averaging
        self.dropout = 0

    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True
        self.input_dim = X.shape[1:]   # (C, H, W)

        flat = _flat_labels(y)
        if self.num_classes is None:
            self.num_classes = max(2, len(np.unique(flat)))
        else:
            self.num_classes = max(2, self.num_classes)

        in_channels = 1 if len(self.input_dim) == 2 else self.input_dim[0]
        self.layers_ = nn.Sequential()

        for i, (k_size, out_ch, stride) in enumerate(zip(self.kernel_sizes, self.channels, self.strides)):
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_ch,
                             kernel_size=k_size, stride=stride,
                             device=self.device)
            if init_:
                nn.init.kaiming_uniform_(conv.weight, a=np.sqrt(5))
            self.layers_.add_module(f'conv_{i}_CW', conv)
            self.layers_.add_module(f'activation_{i}', nn.ReLU())
            in_channels = out_ch

        with torch.no_grad():
            dummy = torch.zeros((1,) + tuple(self.input_dim), device=self.device)
            conv_out = self.layers_(dummy)
            flat_dim = (conv_out.shape[1] if self.global_averaging else int(np.prod(conv_out.shape[1:])))

        self.dense_net = nn.Sequential()
        cur_dim = flat_dim
        for i, h in enumerate(self.hidden_layers):
            self.dense_net.add_module(f'linear_layer_{i}', nn.Linear(cur_dim, h, bias=False, dtype=torch.float, device=self.device))
            self.dense_net.add_module(f'Relu_{i}', nn.ReLU())
            cur_dim = h

        self.final_layer = nn.Linear( cur_dim, 1 if self.problem == 'regression' else self.num_classes, dtype=torch.float, device=self.device)

    def predict(self, X):
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            X = transform_to_torch(arr=X).to(self.device)

        if X.dtype == torch.uint8:
            X = X.float().div_(255.0)
        elif X.dtype != torch.float32:
            X = X.float()

        if X.max() > 2.0:
            X = X / 255.0

        if X.dim() == 2:
            side = int(np.sqrt(X.shape[1]))
            X = X.view(-1, 1, side, side)

        X = self.layers_(X)
        X = X.mean(dim=[2, 3]) if self.global_averaging else X.view(X.size(0), -1)
        prelogits = self.dense_net(X)
        logits = self.final_layer(prelogits)

        return logits.squeeze(-1) if self.problem == 'regression' else logits



class LogisticRegression_Model(NN_model):
    """(Multinomial) logistic regression trained with SGD for tabular data.

    A single linear layer maps the features straight to class logits (or to a
    single scalar for regression). The base trainer's CrossEntropyLoss applies
    the softmax, so this is a plain logistic / softmax regressor with no hidden
    layers. Set ``optimizer='sgd'`` (the default in ``create_model``) to train
    it with stochastic gradient descent.
    """

    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True
        if self.num_classes is None:
            self.num_classes = max(2, len(np.unique(_flat_labels(y))))
        else:
            self.num_classes = max(2, self.num_classes)

        self.input_dim = X.shape[1]
        out_dim = 1 if self.problem == 'regression' else self.num_classes

        self.layers_ = nn.Sequential()
        linear = nn.Linear(self.input_dim, out_dim, bias=bias,
                           device=self.device, dtype=torch.float)
        if init_:
            nn.init.zeros_(linear.weight)
            if bias and linear.bias is not None:
                nn.init.zeros_(linear.bias)
        self.layers_.add_module('logit_layer', linear)



def _group_norm(num_channels, max_groups=32, device=None):
    """GroupNorm with a group count that divides num_channels.

    GroupNorm (not BatchNorm) is used because this data-Shapley pipeline trains
    with very small / single-sample batches (batch_size=1 during value
    estimation), which BatchNorm cannot handle ("Expected more than 1 value per
    channel"). GroupNorm is batch-size independent, so it is robust here.
    """
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, device=device)


class _BasicBlock(nn.Module):
    """Standard two-conv residual block (He et al., 2016).

    Applies two 3x3 convolutions with GroupNorm and a ReLU, plus an identity
    (or 1x1-projection) shortcut that is added back before the final ReLU.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, device=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False, device=device)
        self.bn1 = _group_norm(out_channels, device=device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False, device=device)
        self.bn2 = _group_norm(out_channels, device=device)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False, device=device),
                _group_norm(out_channels, device=device))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class AlexNet_Model(NN_model):
    """AlexNet-STYLE CNN (Krizhevsky et al., 2012)
    """

    def __init__(self, mode_problem='classification', use_img=True,
                 hidden_layers=[1024, 1024], optimizer='sgd', batch_size=128,
                 max_epoch=100, validation_fraction=0.1, early_stopping=0,
                 address=None, test_batch_size=1000, random_seed=666,
                 lr=0.001, num_of_classes=None, normalize=True, device=None,
                 use_fairness=False, fairness_attr=0, num_of_fair=2,
                 lambda_fair=1.0, fair_metric='groupdro', compute_Shapley=True,
                 momentum=0.9, weight_decay=5e-4, lr_schedule='cosine',
                 pixel_mean=0.5, pixel_std=0.5, pool_size=2):
        super().__init__(mode_problem, True, hidden_layers, optimizer,
                         batch_size, max_epoch, validation_fraction,
                         early_stopping, address, test_batch_size,
                         random_seed, lr, num_of_classes, normalize, device,
                         use_fairness, fairness_attr, num_of_fair,
                         lambda_fair, fair_metric, compute_Shapley)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.pool_size = pool_size          # AdaptiveAvgPool2d target (pool_size x pool_size)

    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True
        self.input_dim = X.shape[1:]   # (C, H, W) or (H, W)

        flat = _flat_labels(y)
        if self.num_classes is None:
            self.num_classes = max(2, len(np.unique(flat)))
        else:
            self.num_classes = max(2, self.num_classes)

        in_channels = 1 if len(self.input_dim) == 2 else self.input_dim[0]
        dev = self.device

        def conv_gn_relu(c_in, c_out, k=3, stride=1, pad=1):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=pad,
                          bias=False, device=dev),
                _group_norm(c_out, device=dev),
                nn.ReLU(inplace=True))

        # Stage 1: conv -> pool   (AlexNet conv1 + pool1)
        self.stage1 = nn.Sequential(conv_gn_relu(in_channels, 64),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        # Stage 2: conv -> pool   (AlexNet conv2 + pool2)
        self.stage2 = nn.Sequential(conv_gn_relu(64, 192),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        # Stage 3: conv -> conv -> conv -> pool   (AlexNet conv3/4/5 + pool5)
        self.stage3 = nn.Sequential(conv_gn_relu(192, 384),
                                    conv_gn_relu(384, 256),
                                    conv_gn_relu(256, 256),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.global_pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        flat_dim = 256 * self.pool_size * self.pool_size

        # FC head (AlexNet FC6/FC7/FC8, widths scaled down from 4096->1024)
        self.dense_net = nn.Sequential()
        cur_dim = flat_dim
        for i, h in enumerate(self.hidden_layers):
            self.dense_net.add_module(f'fc_{i}', nn.Linear(cur_dim, h, device=dev, dtype=torch.float))
            self.dense_net.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            cur_dim = h

        self.final_layer = nn.Linear(
            cur_dim, 1 if self.problem == 'regression' else self.num_classes,
            device=dev, dtype=torch.float)

        if init_:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def predict(self, X):
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            X = transform_to_torch(arr=X).to(self.device)

        if X.dtype == torch.uint8:
            X = X.float().div_(255.0)
        elif X.dtype != torch.float32:
            X = X.float()

        if X.max() > 2.0:
            X = X / 255.0

        if X.dim() == 2:
            side = int(np.sqrt(X.shape[1]))
            X = X.view(-1, 1, side, side)
        elif X.dim() == 3:
            X = X.unsqueeze(1)

        pm = getattr(self, 'pixel_mean', None)
        ps = getattr(self, 'pixel_std', None)
        if pm is not None and ps is not None:
            X = (X - pm) / ps

        x = self.stage1(X)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dense_net(x)
        logits = self.final_layer(x)

        return logits.squeeze(-1) if self.problem == 'regression' else logits


class ResNet_Model(NN_model):
    """ResNet backbone trained with SGD for image datasets.

    A CIFAR/MNIST-style residual network: a 3x3 stem, a configurable stack of
    residual stages (``num_blocks`` blocks each, channels doubling and spatial
    size halving per stage after the first), global average pooling, optional
    dense ``hidden_layers``, and a linear classifier. Set ``optimizer='sgd'``
    (the default in ``create_model``) to train it with SGD.
    """

    def __init__(self, mode_problem='classification', hidden_layers=[],
                 optimizer='sgd', batch_size=128, max_epoch=100,
                 validation_fraction=0.1, early_stopping=0, address=None,
                 test_batch_size=1000, random_seed=666, lr=0.001,
                 normalize=True, device=None, base_channels=64,
                 num_blocks=[2, 2, 2, 2], num_of_classes=2,
                 use_fairness=False, fairness_attr=[0], num_of_fair=2,
                 lambda_fair=1.0, fair_metric='groupdro', compute_Shapley=True,
                 momentum=0.9, weight_decay=5e-4, lr_schedule='cosine',
                 pixel_mean=0.5, pixel_std=0.5):

        super().__init__(mode_problem, True, hidden_layers, optimizer,
                         batch_size, max_epoch, validation_fraction,
                         early_stopping, address, test_batch_size,
                         random_seed, lr, num_of_classes, normalize, device,
                         use_fairness, fairness_attr, num_of_fair,
                         lambda_fair, fair_metric, compute_Shapley)
        self.base_channels = base_channels
        self.num_blocks = num_blocks

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def _make_stage(self, out_channels, n_blocks, stride):
        layers = []
        strides = [stride] + [1] * (n_blocks - 1)
        for s in strides:
            layers.append(_BasicBlock(self._in_planes, out_channels,
                                      stride=s, device=self.device))
            self._in_planes = out_channels
        return nn.Sequential(*layers)

    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True
        self.input_dim = X.shape[1:]   # (C, H, W) or (H, W)

        flat = _flat_labels(y)
        if self.num_classes is None:
            self.num_classes = max(2, len(np.unique(flat)))
        else:
            self.num_classes = max(2, self.num_classes)

        in_channels = 1 if len(self.input_dim) == 2 else self.input_dim[0]

        c = self.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1,
                      bias=False, device=self.device),
            _group_norm(c, device=self.device),
            nn.ReLU(inplace=True))

        self._in_planes = c
        stages = []
        for stage_idx, n_blocks in enumerate(self.num_blocks):
            out_c = c * (2 ** stage_idx)
            stride = 1 if stage_idx == 0 else 2
            stages.append(self._make_stage(out_c, n_blocks, stride))
        self.stages = nn.Sequential(*stages)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense_net = nn.Sequential()
        cur_dim = self._in_planes
        for i, h in enumerate(self.hidden_layers):
            self.dense_net.add_module(f'linear_{i}', nn.Linear(
                cur_dim, h, device=self.device, dtype=torch.float))
            self.dense_net.add_module(f'relu_{i}', nn.ReLU())
            cur_dim = h

        self.final_layer = nn.Linear(
            cur_dim, 1 if self.problem == 'regression' else self.num_classes,
            device=self.device, dtype=torch.float)

        if init_:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def predict(self, X):
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            X = transform_to_torch(arr=X).to(self.device)

        if X.dtype == torch.uint8:
            X = X.float().div_(255.0)
        elif X.dtype != torch.float32:
            X = X.float()

        if X.max() > 2.0:
            X = X / 255.0

        if X.dim() == 2:
            side = int(np.sqrt(X.shape[1]))
            X = X.view(-1, 1, side, side)
        elif X.dim() == 3:
            # (N, H, W) -> add channel dim
            X = X.unsqueeze(1)

        pm = getattr(self, 'pixel_mean', None)
        ps = getattr(self, 'pixel_std', None)
        if pm is not None and ps is not None:
            X = (X - pm) / ps

        x = self.stem(X)
        x = self.stages(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dense_net(x)
        logits = self.final_layer(x)

        return logits.squeeze(-1) if self.problem == 'regression' else logits



class _SoftTree(nn.Module):
    """A single differentiable ("soft") decision tree of fixed depth."""

    def __init__(self, in_dim, depth, out_dim, device=None, init_=True):
        super().__init__()
        self.depth = depth
        self.level_linears = nn.ModuleList([
            nn.Linear(in_dim, 2 ** d, device=device, dtype=torch.float)
            for d in range(depth)])
        self.leaf_logits = nn.Parameter(
            torch.zeros(2 ** depth, out_dim, device=device, dtype=torch.float))
        if init_:
            for lin in self.level_linears:
                nn.init.kaiming_uniform_(lin.weight, a=np.sqrt(5))
            nn.init.normal_(self.leaf_logits, std=0.01)

    def forward(self, X):
        probs = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
        for lin in self.level_linears:
            gate = torch.sigmoid(lin(X))            # P(go right) per node at this level
            left = probs * (1 - gate)
            right = probs * gate
            probs = torch.stack([left, right], dim=2).reshape(X.shape[0], -1)
        return probs @ self.leaf_logits              # (batch, out_dim)


class SoftDecisionTree_Model(NN_model):
    """Differentiable ("soft"/gradient) decision tree trained with SGD."""

    def __init__(self, mode_problem='classification', use_img=False,
                 hidden_layers=[100], optimizer='sgd', batch_size=128,
                 max_epoch=100, validation_fraction=0.1, early_stopping=15,
                 address=None, test_batch_size=1000, random_seed=666,
                 lr=0.001, num_of_classes=None, normalize=True, device=None,
                 use_fairness=False, fairness_attr=0, num_of_fair=2,
                 lambda_fair=1.0, fair_metric='groupdro', compute_Shapley=True,
                 depth=4):
        super().__init__(mode_problem, use_img, hidden_layers, optimizer,
                         batch_size, max_epoch, validation_fraction,
                         early_stopping, address, test_batch_size,
                         random_seed, lr, num_of_classes, normalize, device,
                         use_fairness, fairness_attr, num_of_fair,
                         lambda_fair, fair_metric, compute_Shapley)
        self.depth = depth

    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True
        if self.num_classes is None:
            self.num_classes = max(2, len(np.unique(_flat_labels(y))))
        else:
            self.num_classes = max(2, self.num_classes)

        self.input_dim = X.shape[1]
        out_dim = 1 if self.problem == 'regression' else self.num_classes
        self.tree = _SoftTree(self.input_dim, self.depth, out_dim,
                              device=self.device, init_=init_)

    def predict(self, X):
        logits = self.tree(X)
        return logits.squeeze(-1) if self.problem == 'regression' else logits


class BoostedTree_Model(NN_model):
    """Additive ensemble of differentiable soft decision trees ("boosted
    tree"), trained jointly with SGD."""

    def __init__(self, mode_problem='classification', use_img=False,
                 hidden_layers=[100], optimizer='sgd', batch_size=128,
                 max_epoch=100, validation_fraction=0.1, early_stopping=15,
                 address=None, test_batch_size=1000, random_seed=666,
                 lr=0.001, num_of_classes=None, normalize=True, device=None,
                 use_fairness=False, fairness_attr=0, num_of_fair=2,
                 lambda_fair=1.0, fair_metric='groupdro', compute_Shapley=True,
                 num_trees=10, tree_depth=3, shrinkage=None):
        super().__init__(mode_problem, use_img, hidden_layers, optimizer,
                         batch_size, max_epoch, validation_fraction,
                         early_stopping, address, test_batch_size,
                         random_seed, lr, num_of_classes, normalize, device,
                         use_fairness, fairness_attr, num_of_fair,
                         lambda_fair, fair_metric, compute_Shapley)
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.shrinkage = shrinkage

    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True
        if self.num_classes is None:
            self.num_classes = max(2, len(np.unique(_flat_labels(y))))
        else:
            self.num_classes = max(2, self.num_classes)

        self.input_dim = X.shape[1]
        out_dim = 1 if self.problem == 'regression' else self.num_classes
        self.trees = nn.ModuleList([
            _SoftTree(self.input_dim, self.tree_depth, out_dim,
                     device=self.device, init_=init_)
            for _ in range(self.num_trees)])
        self._shrink = (self.shrinkage if self.shrinkage is not None
                        else 1.0 / self.num_trees)

    def predict(self, X):
        logits = sum(t(X) for t in self.trees) * self._shrink
        return logits.squeeze(-1) if self.problem == 'regression' else logits


class _ObliviousTree(nn.Module):
    """A single "oblivious" differentiable decision tree, as used by NODE
    (Popov, Morozov & Babenko, "Neural Oblivious Decision Ensembles for Deep
    Learning on Tabular Data", ICLR 2020).
    """

    def __init__(self, in_dim, depth, out_dim, device=None, init_=True):
        super().__init__()
        self.depth = depth
        self.feature_weights = nn.Parameter(
            torch.zeros(depth, in_dim, device=device, dtype=torch.float))
        self.thresholds = nn.Parameter(
            torch.zeros(depth, device=device, dtype=torch.float))
        self.log_temperatures = nn.Parameter(
            torch.zeros(depth, device=device, dtype=torch.float))
        self.response = nn.Parameter(
            torch.zeros(2 ** depth, out_dim, device=device, dtype=torch.float))
        if init_:
            nn.init.normal_(self.feature_weights, std=0.1)
            nn.init.normal_(self.thresholds, std=0.1)
            nn.init.normal_(self.response, std=0.01)

    def forward(self, X):
        batch = X.shape[0]
        probs = torch.ones(batch, 1, device=X.device, dtype=X.dtype)
        for d in range(self.depth):
            feat_w = F.softmax(self.feature_weights[d], dim=0)   # soft feature selection
            x_sel = X @ feat_w                                    # (batch,) selected value
            tau = F.softplus(self.log_temperatures[d]) + 1e-3     # always positive
            gate = torch.sigmoid((x_sel - self.thresholds[d]) / tau)
            gate = gate.unsqueeze(1)                              # (batch, 1) -- SAME split
            left = probs * (1 - gate)                             # applied to every node
            right = probs * gate                                  # at this level (oblivious)
            probs = torch.stack([left, right], dim=2).reshape(batch, -1)
        return probs @ self.response                              # (batch, out_dim)


class NODE_Model(NN_model):
    """NODE: Neural Oblivious Decision Ensembles (Popov, Morozov & Babenko,
    ICLR 2020), trained end-to-end with standard SGD/Adam.
    """

    def __init__(self, mode_problem='classification', use_img=False,
                 hidden_layers=[100], optimizer='sgd', batch_size=128,
                 max_epoch=100, validation_fraction=0.1, early_stopping=15,
                 address=None, test_batch_size=1000, random_seed=666,
                 lr=0.001, num_of_classes=None, normalize=True, device=None,
                 use_fairness=False, fairness_attr=0, num_of_fair=2,
                 lambda_fair=1.0, fair_metric='groupdro', compute_Shapley=True,
                 num_layers=2, trees_per_layer=8, depth=4):
        super().__init__(mode_problem, use_img, hidden_layers, optimizer,
                         batch_size, max_epoch, validation_fraction,
                         early_stopping, address, test_batch_size,
                         random_seed, lr, num_of_classes, normalize, device,
                         use_fairness, fairness_attr, num_of_fair,
                         lambda_fair, fair_metric, compute_Shapley)
        self.num_layers = num_layers
        self.trees_per_layer = trees_per_layer
        self.depth = depth

    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True
        if self.num_classes is None:
            self.num_classes = max(2, len(np.unique(_flat_labels(y))))
        else:
            self.num_classes = max(2, self.num_classes)

        self.input_dim = X.shape[1]
        out_dim = 1 if self.problem == 'regression' else self.num_classes
        self._out_dim = out_dim

        self.node_layers = nn.ModuleList()
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layer_trees = nn.ModuleList([
                _ObliviousTree(in_dim, self.depth, out_dim,
                              device=self.device, init_=init_)
                for _ in range(self.trees_per_layer)])
            self.node_layers.append(layer_trees)
            in_dim = in_dim + self.trees_per_layer * out_dim

    def predict(self, X):
        h = X
        all_outputs = []
        for layer_trees in self.node_layers:
            layer_out = torch.stack([t(h) for t in layer_trees], dim=0)  # (trees, batch, out)
            all_outputs.append(layer_out)
            concat_out = layer_out.permute(1, 0, 2).reshape(X.shape[0], -1)
            h = torch.cat([h, concat_out], dim=1)
        logits = torch.cat(all_outputs, dim=0).mean(dim=0)                # (batch, out_dim)
        return logits.squeeze(-1) if self.problem == 'regression' else logits