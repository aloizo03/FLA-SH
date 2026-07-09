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

    def __init__(self, mode_problem='classification', use_img=False,
                 hidden_layers=[100], optimizer='sgd', batch_size=128,
                 max_epoch=100, validation_fraction=0.1, early_stopping=15,
                 address=None, test_batch_size=1000, random_seed=666,
                 lr=0.001, num_of_classes=None, normalize=True, device=None,
                 use_fairness=False, fairness_attr=0, num_of_fair=2,
                 lambda_fair=1.0, fair_metric='groupdro', compute_Shapley=True):
        super().__init__()
        self.problem = mode_problem
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = max_epoch
        self.val_fraction = validation_fraction
        self.early_stop = early_stopping
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

        self._label_map = None
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
            show_progr=False, fair_attrbs=None, fair_attrbs_val=None):

        all_y = _flat_labels(y)
        if self.num_classes is None:
            self.num_classes = max(2, len(np.unique(all_y)))

        if self.problem == 'classification':
            self._build_label_map(all_y)
            unique_labels = sorted(set(int(v) for v in all_y))
            required_classes = max(unique_labels) + 1  # must cover max label
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
            if isinstance(X, list):
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

        fair_train = None
        fair_val = None

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
                fair_val = []

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
                        fair_train.append(ft);
                        fair_val.append(fv)

                elif isinstance(X, list) and isinstance(y, list):
                    fa_list = (
                        fair_attrbs if (self.use_fairness and isinstance(fair_attrbs, list)) else [None] * len(X))
                    for X_, y_, fa_ in zip(X, y, fa_list):
                        n = X_.shape[0] if isinstance(X_, np.ndarray) else len(X_)
                        if n < min_split:
                            Xt, Xv, yt, yv, ft, fv = self._small_chunk_val_split(X_, y_, fa_)
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
                    X_train_lst, X_val_lst, y_train_lst, y_val_lst, fair_train, fair_val = train_test_split(X, y,
                                                                                                            fair_attrbs,
                                                                                                            test_size=self.val_fraction)
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
                       data_dict=data_dict, show_progr=show_progr)

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

        out_layer = nn.Linear(hidden_dim, 1 if self.problem == 'regression' else self.num_classes, bias=bias,
                              dtype=torch.float, device=self.device)
        if init_:
            nn.init.kaiming_uniform_(out_layer.weight, a=np.sqrt(5))
        self.layers_.add_module('out_layer', out_layer)

    def training_(self, X, y, X_val, y_val, fair_train, fair_val,
                  use_val, max_epochs, batch_size, data_dict, show_progr):
        assert len(X) == len(y), 'Input and labels not the same size'

        self.history = {'metrics': [], 'idxs': [], 'keys': []}
        stop_counter = 0
        higher_is_better = self.metric in ['accuracy', 'auc', 'f1']
        best_performance = -np.inf if higher_is_better else np.inf

        if self.optimizer.lower() == 'sgd':
            optimiser = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.normalize:
            if isinstance(X, list):
                if not self.use_img:
                    X = [preprocessing.normalize(a) if a is not None else None for a in X]
                    if X_val is not None:
                        if isinstance(X_val, list):
                            X_val = [preprocessing.normalize(a) if a is not None else None for a in X_val]
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

        for epoch in range(max_epochs):
            idxs, val_metrics, keys_lst = self.one_epoch_iter(
                X=X, y=y, X_val=X_val, y_val=y_val,
                use_val=use_val, fair_train=fair_train, fair_val=fair_val,
                batch_size=batch_size, data_dict=data_dict, optimiser=optimiser)

            self.history['metrics'] = val_metrics
            self.history['idxs'] = idxs
            self.history['keys'] = keys_lst

            current_performance = (np.mean(val_metrics) if (use_val and val_metrics) else float('nan'))

            if epoch % 10 == 0 and show_progr:
                print(f'Epoch {epoch}/{max_epochs}, '
                      f'avg_val_metric: {current_performance:.4f}, '
                      f'best: {best_performance:.4f}')

            if self.early_stop > 0 and X_val is not None and val_metrics:
                improved = (current_performance > best_performance
                            if higher_is_better
                            else current_performance < best_performance)
                if improved:
                    best_performance = current_performance
                    stop_counter = 0
                else:
                    stop_counter += 1
                    if stop_counter > self.early_stop:
                        print(f'Early stopping at epoch {epoch}. '
                              f'Best: {best_performance:.4f}')
                        break

    def one_epoch_iter(self, X, y, X_val, y_val, fair_train, fair_val,
                       use_val, batch_size, optimiser, data_dict=None):
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
                batches = self._make_tabular_batches(X, y, X_val, y_val, fair_train, fair_val, use_val, batch_size,
                                                     idxs)
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

        if self.use_fairness:
            loss_fair_fn = (GroupDRO(n_groups=self.num_of_fair, device=self.device)
                            if self.fair_metric == 'groupdro' else None)
        else:
            loss_fair_fn = None

        if self.problem == "classification":
            loss_fn = nn.CrossEntropyLoss(reduction='mean').to(self.device)
            loss_fn_none = nn.CrossEntropyLoss(reduction='none').to(self.device)
        else:
            loss_fn = nn.MSELoss(reduction='mean').to(self.device)
            loss_fn_none = nn.MSELoss(reduction='none').to(self.device)

        for batch in batches:
            (key_, X_batch, y_batch, X_val_batch, y_val_batch, fair_train_batch, fair_val_batch) = batch

            self.train()

            X_batch = transform_to_torch(X_batch).to(self.device)
            y_batch = transform_to_torch(y_batch).to(self.device).ravel()

            if X_batch.ndim == 1:
                X_batch = X_batch.unsqueeze(0)
            if self.use_img and X_batch.ndim == 3:
                X_batch = X_batch.unsqueeze(0)

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
            y_pred = self.predict(X_batch)
            per_sample_losses = None

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
                    loss = loss + self.lambda_fair * variance_group_loss(groups=fair_train_batch,
                                                                         per_sample_losses=per_sample_losses,
                                                                         lambda_var=1)

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
                fvb = _fitem(fair_val, k)
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
            fv = _s(fair_val, sl) if fair_val is not None else None
            batches.append((bi, _s(X, sl), _s(y, sl), Xv, yv, ft, fv))
        return batches

    def validation(self, X, y, fair_val, loss_fn, loss_fair_fn):
        self.eval()
        with torch.no_grad():
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
                    metric_val -= (
                                self.lambda_fair * group_dro_loss(loss_fair_fn, fair_val, val_loss_per_sample)).item()
                elif self.fair_metric == 'cvar':
                    metric_val -= cvar_loss(per_sample_losses=val_loss_per_sample).item()
                elif self.fair_metric == 'varianceloss':
                    metric_val -= variance_group_loss(groups=fair_val, per_sample_losses=val_loss_per_sample,
                                                      lambda_var=1).item()

            return metric_val

    def predict(self, X):
        return self.layers_(X)


class LSTM_Model(NN_model):

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

        embedding_dim = self.hidden_layers[0] if self.hidden_layers else 64
        hidden_dim = self.hidden_layers[-1] if self.hidden_layers else 64

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0,
                                      device=self.device)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, device=self.device)
        self.out_layer = nn.Linear(hidden_dim, 1 if self.problem == 'regression' else self.num_classes, bias=bias,
                                   dtype=torch.float, device=self.device)

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
        self.input_dim = X.shape[1:]  # (C, H, W)

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
            self.dense_net.add_module(f'linear_layer_{i}',
                                      nn.Linear(cur_dim, h, bias=False, dtype=torch.float, device=self.device))
            self.dense_net.add_module(f'Relu_{i}', nn.ReLU())
            cur_dim = h

        self.final_layer = nn.Linear(cur_dim, 1 if self.problem == 'regression' else self.num_classes,
                                     dtype=torch.float, device=self.device)

    def predict(self, X):
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            X = transform_to_torch(arr=X).to(self.device)

        if X.dtype == torch.uint8:
            X = X.float().div_(255.0)
        elif X.dtype != torch.float32:
            X = X.float()

        if X.dim() == 2:
            side = int(np.sqrt(X.shape[1]))
            X = X.view(-1, 1, side, side)

        X = self.layers_(X)
        X = X.mean(dim=[2, 3]) if self.global_averaging else X.view(X.size(0), -1)
        prelogits = self.dense_net(X)
        logits = self.final_layer(prelogits)

        return logits.squeeze(-1) if self.problem == 'regression' else logits