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

from utils import transform_to_torch, get_metric, GroupDRO, group_dro_loss


class NN_model(nn.Module):

    def __init__(self, mode_problem='classification', use_img=False, hidden_layers=[100], optimizer='sgd', batch_size=128, max_epoch=100,  validation_fraction=0.1,
                 early_stopping=15, address=None, test_batch_size=1000, random_seed=666, lr=0.001, num_of_classes=None, normalize=True, device=None, use_fairness=False, fairness_attr=0, num_of_fair=2, lambda_fair=1.0):
        super().__init__()
        self.problem = mode_problem
        self.hidden_layers=hidden_layers
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
        self.use_img=use_img
        self.num_classes = num_of_classes

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if self.problem.lower() != 'classification' and self.problem.lower() != 'regression':
            raise ValueError('Problem must be either classification or regression')
        self.is_built = False

        self.use_fairness = use_fairness
        self.fairness_attr = fairness_attr
        self.num_of_fair = num_of_fair
        self.lambda_fair = lambda_fair


    def fit(self, X, y, X_val, y_val, use_val=True, epochs=None, batch_size=None, save=False, load=False, sample_weight=None, data_dict=None, use_chunks=True, metric='accuracy', show_progr=False, fair_attrbs=None, fair_attrbs_val=None):

        if isinstance(y, list):
            # all_y = pd.concat(y, axis=0).to_numpy().ravel().tolist()
            if all(isinstance(item, pd.DataFrame) for item in y):
                all_y = pd.concat(y, axis=0)
                if self.num_classes is None:
                    self.num_classes = len(set(all_y))
            elif all(isinstance(item, np.ndarray) for item in y):
                all_y = np.concatenate(y, axis=0)
                if self.num_classes is None:
                    unique_values = np.unique(all_y)
                    self.num_classes = unique_values.shape[0]
        elif isinstance(y, pd.DataFrame):
            all_y = y.to_numpy().ravel().tolist()
            if self.num_classes is None:
                self.num_classes = len(set(all_y))
        else:
            if self.num_classes is None:
                all_y = y.ravel().tolist()
                self.num_classes = len(set(all_y))

        self.metric = metric

        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
        if not self.is_built:
            if isinstance(X, list): 
                if all(isinstance(item, pd.DataFrame) for item in X):
                    all_X = pd.concat(X, axis=0).to_numpy()
                elif all(isinstance(item, np.ndarray) for item in X):
                    all_X = np.concatenate(X, axis=0)
            elif isinstance(X, pd.DataFrame):
                all_X = X.to_numpy()
            else:
                all_X = X

            if isinstance(y, list): 
                if all(isinstance(item, pd.DataFrame) for item in y):
                    all_y = pd.concat(y, axis=0).to_numpy().ravel().tolist()
                elif all(isinstance(item, np.ndarray) for item in y):
                    all_y = np.concatenate(y, axis=0).ravel().tolist()
            elif isinstance(y, np.ndarray):
                all_y = y
            else:
                all_y = y.to_numpy()   
            
            self.build(all_X, all_y)
        
        fair_train = None
        fair_val = None

        if len(X) > 0 and not X is None:
            if use_chunks:
                if X_val is None and use_val:
                    X_train_lst = []
                    X_val_lst = []
                    y_train_lst = []
                    y_val_lst = []
                    fair_train = []
                    fair_val = []
                    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
                        if self.use_fairness is not True:
                            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_fraction)
                        else:
                           X_train, X_val, y_train, y_val, fair_attrbs_train, fair_attrb_val = train_test_split(X, y, fair_attrbs, test_size=self.val_fraction)
                           fair_train.append(fair_attrbs_train)
                           fair_val.append(fair_attrb_val)
                        X_train_lst.append(X_train)

                        X_val_lst.append(X_val)
                        y_train_lst.append(y_train)
                        y_val_lst.append(y_val)
                    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                        if self.use_fairness is not True:
                            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_fraction)
                        else:
                           X_train, X_val, y_train, y_val, fair_attrbs_train, fair_attrb_val = train_test_split(X, y, fair_attrbs, test_size=self.val_fraction)
                           fair_train.append(fair_attrbs_train)
                           fair_val.append(fair_attrb_val)

                        X_train_lst.append(X_train)
                        X_val_lst.append(X_val)
                        y_train_lst.append(y_train)
                        y_val_lst.append(y_val)
                    elif isinstance(X, list) and isinstance(y, list):
                        if self.use_fairness is not True:
                            for X_, y_ in zip(X, y):
                                X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=self.val_fraction)

                                X_train_lst.append(X_train)
                                X_val_lst.append(X_val)
                                y_train_lst.append(y_train)
                                y_val_lst.append(y_val)
                        else:
                            if isinstance(fair_attrbs, list):
                                for X_, y_, fair_attrb in zip(X, y, fair_attrbs):
                                    X_train, X_val, y_train, y_val, fair_attrbs_train, fair_attrb_val = train_test_split(X_, y_, fair_attrb, test_size=self.val_fraction)
                                    X_train_lst.append(X_train)
                                    X_val_lst.append(X_val)
                                    y_train_lst.append(y_train)
                                    y_val_lst.append(y_val)

                                    fair_train.append(fair_attrbs_train)
                                    fair_val.append(fair_attrb_val)
                else:
                    if self.use_fairness:
                        fair_train = fair_attrbs
                        fair_val = fair_attrbs_val
                    X_train_lst, y_train_lst, X_val_lst, y_val_lst = X, y, X_val, y_val
            else:
                if X_val is None and self.val_fraction * len(X) > 2 and use_val:
                    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
                        if self.use_fairness:
                            X_train_lst, X_val_lst, y_train_lst, y_val_lst, fair_train, fair_val = train_test_split(X, y, fair_attrbs, test_size=self.val_fraction)
                        else:
                            X_train_lst, X_val_lst, y_train_lst, y_val_lst = train_test_split(X, y, test_size=self.val_fraction)
                        # X_train_lst, y_train_lst = X_np, y_np

                    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                        if self.use_fairness:
                            # if isinstance(fair_attrbs, pd.DataFrame):
                            #     fair_attrbs = 
                            X_train_lst, X_val_lst, y_train_lst, y_val_lst, fair_train, fair_val= train_test_split(X, y, fair_attrbs, test_size=self.val_fraction)
                        else:
                            X_train_lst, X_val_lst, y_train_lst, y_val_lst = train_test_split(X, y, test_size=self.val_fraction)
                        # X_train_lst, y_train_lst = X, y
                    elif isinstance(X, list) and isinstance(y, list):
                        if self.use_fairness:
                            X_train_lst, X_val_lst, y_train_lst, y_val_lst, fair_train, fair_val = train_test_split(X, y, fair_attrbs, test_size=self.val_fraction)
                        else:
                            X_train_lst, X_val_lst, y_train_lst, y_val_lst = train_test_split(X, y, test_size=self.val_fraction)
                        # X_train_lst, y_train_lst = X_np, y_np
                else:
                    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
                        X = X.to_numpy()
                        y = y.to_numpy()
                    if isinstance(X_val, pd.DataFrame) and isinstance(y_val, pd.DataFrame):
                        X_val = X_val.to_numpy()
                        y_val = y_val.to_numpy()
                    X_train_lst, y_train_lst, X_val_lst, y_val_lst = X, y, X_val, y_val
                    if self.use_fairness:
                        fair_train = fair_attrbs
                        fair_val = fair_attrbs_val
            self.training_(X=X_train_lst, y=y_train_lst, X_val=X_val_lst, y_val=y_val_lst, fair_train=fair_train, fair_val=fair_val, use_val=use_val, max_epochs=epochs, batch_size=batch_size, data_dict=data_dict, show_progr=show_progr)

    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True
        if isinstance(y, np.ndarray):
            y = y.ravel().tolist()
        else:
            if self.num_classes is None:
                self.num_classes = len(set(y))
        self.input_dim = X.shape[1]
        self.layers_ = nn.Sequential()

        hidden_dim = self.hidden_layers[0]
        input_layer = nn.Linear(self.input_dim, hidden_dim, bias=bias, device=self.device, dtype=torch.float)
        if init_:
            nn.init.kaiming_uniform_(input_layer.weight, a=np.sqrt(5))

        self.layers_.add_module('linear_0', input_layer)
        self.layers_.add_module('activation_0', nn.ReLU())

        for i, layer_dim in enumerate(self.hidden_layers):
            linear = nn.Linear(hidden_dim, layer_dim, bias=bias,
                               device=self.device, dtype=torch.float)
            hidden_dim = layer_dim
            if init_:
                nn.init.kaiming_uniform_(linear.weight, a=np.sqrt(5))
            self.layers_.add_module(f'linear_{i + 1}', linear)
            self.layers_.add_module(f'activation_{i + 1}', nn.ReLU())

        out_layer = nn.Linear(hidden_dim, 1 if self.problem == 'regression' else self.num_classes, bias=bias, dtype=torch.float, device=self.device)

        if init_:
                nn.init.kaiming_uniform_(out_layer.weight, a=np.sqrt(5))
        
        self.layers_.add_module('out_layer', out_layer)

    def training_(self, X, y, X_val, y_val, fair_train, fair_val, use_val, max_epochs, batch_size, data_dict, show_progr):
        assert len(X)==len(y), 'Input and labels not the same size'
        self.history = {'metrics':[], 'idxs':[], 'keys':[]}
        stop_counter = 0
        best_performance = -np.inf if self.metric in ['accuracy', 'auc', 'f1'] else np.inf

        optimiser = torch.optim.SGD(self.parameters(), lr=self.lr) if self.optimizer.lower() == 'sgd' else torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.normalize:
            if isinstance(X, list):
                if self.use_img:
                    X = [x.astype(np.float32) / 255.0 for x in X]
                else:
                    # Prefer standardization across features
                    X = [preprocessing.normalize(arr) for arr in X]
                if X_val is not None:
                    if self.use_img:
                        X_val = [x.astype(np.float32) / 255.0 for x in X_val]
                    else:
                        X_val = [preprocessing.normalize(arr)  for arr in X_val]
            elif isinstance(X, np.ndarray):
                if self.use_img:
                    X = X.astype(np.float32) / 255.0
                    if X_val is not None:
                        X_val = X_val.astype(np.float32) / 255.0
                else:
                    X = preprocessing.normalize(X)
                    if X_val is not None:
                        X_val = preprocessing.normalize(X_val)
                            
        for epoch in range(max_epochs):
            idxs, val_metrics, keys_lst = self.one_epoch_iter(X=X, y=y, X_val=X_val, y_val=y_val, use_val=use_val, fair_train=fair_train, fair_val=fair_val, batch_size=batch_size, data_dict=data_dict, optimiser=optimiser)
            self.history['metrics'] = val_metrics
            self.history['idxs'] = idxs
            self.history['keys'] = keys_lst
            
            if epoch % 10 == 0 and show_progr:
                print(f'Epoch {epoch}/{max_epochs}, average_val_loss: {np.mean(val_metrics)}, best_perf: {best_performance}')
            if self.early_stop > 0 and X_val is not None:
                current_performance = np.mean(val_metrics)
                if best_performance is None:
                    best_performance = current_performance
                if current_performance > best_performance:
                    best_performance = current_performance
                    stop_counter = 0
                else:
                    stop_counter += 1
                    if stop_counter > self.early_stop:
                        print(f'Finishing at epoch {epoch} because at erply stoping at the last {stop_counter} epochs with best performance {best_performance}')
                        break
            
    def one_epoch_iter(self, X, y, X_val, y_val, fair_train, fair_val, use_val, batch_size, optimiser, data_dict=None,):
        val_metrics = []
        keys_lst = []
        if self.use_img:
            # Todo: fairness for images
            if data_dict is None:
                idxs = np.random.permutation(len(X))
                
                if isinstance(X_val, np.ndarray) or isinstance(X_val, pd.DataFrame):
                    if self.use_fairness:
                        if isinstance(fair_train, pd.DataFrame):
                            # fair_train is sliced, fair_val is passed whole (matching X_val)
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], X_val, y_val, fair_train.iloc[k*batch_size : (k+1)*batch_size], fair_val) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], X_val, y_val, fair_train[k*batch_size : (k+1)*batch_size], fair_val) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    else:
                        batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], X_val, y_val, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                
                elif isinstance(X_val, list) and isinstance(y_val, list):
                    if self.use_fairness:
                        if isinstance(fair_train, pd.DataFrame):
                            # fair_train is sliced, fair_val is sliced (matching X_val)
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], X_val[k*batch_size : (k+1)*batch_size], y_val[k*batch_size : (k+1)*batch_size], fair_train.iloc[k*batch_size : (k+1)*batch_size], fair_val.iloc[k*batch_size : (k+1)*batch_size]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], X_val[k*batch_size : (k+1)*batch_size], y_val[k*batch_size : (k+1)*batch_size], fair_train[k*batch_size : (k+1)*batch_size], fair_val[k*batch_size : (k+1)*batch_size]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    else:
                        batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], X_val[k*batch_size : (k+1)*batch_size], y_val[k*batch_size : (k+1)*batch_size], None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                
                elif use_val is not True:
                    if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame):
                        if self.use_fairness:
                            if isinstance(fair_train, pd.DataFrame):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], None, None, fair_train.iloc[k*batch_size : (k+1)*batch_size], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], None, None, fair_train[k*batch_size : (k+1)*batch_size], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], None, None, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, list) and isinstance(y, list):
                        if self.use_fairness:
                            if isinstance(fair_train, pd.DataFrame):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], None, None, fair_train.iloc[k*batch_size : (k+1)*batch_size], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], None, None, fair_train[k*batch_size : (k+1)*batch_size], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k*batch_size : (k+1)*batch_size], y[k*batch_size : (k+1)*batch_size], None, None, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
            else:
                idxs = np.random.permutation(len(data_dict.keys()))
                keys_lst = list(data_dict.keys())    
        
                batches = [(keys_lst[int(i)], X[i], y[i], X_val[i], y_val[i]) for i in idxs]
        else:

            if data_dict is None:
                idxs = np.random.permutation(len(X))
                if isinstance(X_val, np.ndarray) or isinstance(X_val, pd.DataFrame):
                    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val, y_val, fair_train.iloc[k], fair_val.iloc[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val, y_val, fair_train[k], fair_val[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val, y_val, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y.iloc[k], X_val, y_val, fair_train.iloc[k], fair_val.iloc[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y.iloc[k], X_val, y_val, fair_train[k], fair_val[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y.iloc[k], X_val, y_val, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y[k], X_val, y_val, fair_train.iloc[k], fair_val.iloc[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y[k], X_val, y_val, fair_train[k], fair_val[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y[k], X_val, y_val, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, np.ndarray) and isinstance(y, pd.DataFrame):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y.iloc[k], X_val, y_val, fair_train.iloc[k], fair_val.iloc[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y.iloc[k], X_val, y_val, fair_train[k], fair_val[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y.iloc[k], X_val, y_val, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, list) and isinstance(y, list):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val, y_val, fair_train.iloc[k], fair_val.iloc[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val, y_val, fair_train[k], fair_val[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val, y_val, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]

                # When val is list
                elif isinstance(X_val, list) and isinstance(y_val, list):
                    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val, y_val, fair_train.iloc[k], fair_val.iloc[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val, y_val, fair_train[k], fair_val[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val, y_val, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y.iloc[k], X_val, y_val, fair_train.iloc[k], fair_val.iloc[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y.iloc[k], X_val, y_val, fair_train[k], fair_val[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y.iloc[k], X_val, y_val, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y[k], X_val, y_val, fair_train.iloc[k], fair_val.iloc[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y[k], X_val, y_val, fair_train[k], fair_val[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y[k], X_val, y_val, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, np.ndarray) and isinstance(y, pd.DataFrame):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y.iloc[k], X_val, y_val, fair_train.iloc[k], fair_val.iloc[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y.iloc[k], X_val, y_val, fair_train[k], fair_val[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y.iloc[k], X_val, y_val, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, list) and isinstance(y, list):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val[k], y_val[k], fair_train.iloc[k], fair_val.iloc[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val[k], y_val[k], fair_train[k], fair_val[k]) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], X_val[k], y_val[k], None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]   
                #  WHen validation is None
                else:
                    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], None, None, fair_train.iloc[k], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], None, None, fair_train[k], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], None, None, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y.iloc[k], None, None, fair_train.iloc[k], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y.iloc[k], None, None, fair_train[k], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y.iloc[k], None, None, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y[k], None, None, fair_train.iloc[k], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y[k], None, None, fair_train[k], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X.iloc[k], y[k], None, None, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, np.ndarray) and isinstance(y, pd.DataFrame):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y.iloc[k], None, None, fair_train.iloc[k], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y.iloc[k], None, None, fair_train[k], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y.iloc[k], None, None, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                    elif isinstance(X, list) and isinstance(y, list):
                        if self.use_fairness:
                            if isinstance(fair_train, (pd.DataFrame, pd.Series)):
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], None, None, fair_train.iloc[k], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                            else:
                                batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], None, None, fair_train[k], None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
                        else:
                            batches = [(idxs[k*batch_size : (k+1)*batch_size], X[k], y[k], None, None, None, None) for k in range(int(np.ceil(len(idxs)/batch_size)))]
            else:
                idxs = np.random.permutation(len(data_dict.keys()))
                keys_lst = list(data_dict.keys())    
                if self.use_fairness:
                    batches = [(keys_lst[int(i)], X[i], y[i], X_val[i], y_val[i], fair_train[i], fair_val[i]) for i in idxs]
                else:
                    batches = [(keys_lst[int(i)], X[i], y[i], X_val[i], y_val[i], None, None) for i in idxs]

        if self.use_fairness:
            loss_fair_fn = GroupDRO(n_groups=self.num_of_fair, device=self.device)

        if self.problem == "classification":
            loss_fn = nn.CrossEntropyLoss(reduction='mean').to(self.device)
            loss_fn_none = nn.CrossEntropyLoss(reduction='none').to(self.device)
        elif self.problem == "regression":
            loss_fn = nn.MSELoss(reduction='mean').to(self.device)
            loss_fn_none = nn.MSELoss(reduction='none').to(self.device)

        for idx, batch in enumerate(batches):
            key_, X_batch, y_batch, X_val_batch, y_val_batch, fair_train_batch, fair_val_batch = batch
            self.train()

            X_batch = transform_to_torch(X_batch).to(self.device)
            y_batch = transform_to_torch(y_batch).to(self.device)
            y_batch = y_batch.ravel()

            if X_batch.ndim == 1:
                X_batch = X_batch.unsqueeze(0)
            
            if self.use_img and X_batch.ndim == 3:
                X_batch = X_batch.unsqueeze(0)

            if self.use_fairness:
                fair_train_batch = transform_to_torch(fair_train_batch).to(self.device).long()
                if use_val:
                    fair_val_batch = transform_to_torch(fair_val_batch).to(self.device).long()
            
            if use_val:
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
                dro_term = group_dro_loss(loss_fair_fn, fair_train_batch, per_sample_losses)
                loss = loss + self.lambda_fair * dro_term
                        
            loss.backward()
            optimiser.step()
            
            keys_lst.append(key_)

            if use_val:
                if self.use_fairness:
                    val_metric = self.validation(X=X_val_batch, y=y_val_batch, fair_val=fair_val_batch, loss_fn=loss_fn_none, loss_fair_fn=loss_fair_fn)
                else:
                    val_metric = self.validation(X=X_val_batch, y=y_val_batch, fair_val=fair_val_batch, loss_fn=loss_fn_none, loss_fair_fn=None)
                val_metrics.append(val_metric)

        return idxs, val_metrics, keys_lst

    def validation(self, X, y, fair_val, loss_fn, loss_fair_fn,):
        self.eval()
        with torch.no_grad():
            y_pred = self.predict(X) 
            
            if self.problem == "classification":
                y = y.long()
                y_pred_max = torch.argmax(y_pred, dim=1)
                y_pred_metric = y_pred_max.cpu().numpy()
                
                # Use logits for loss
                val_loss_per_sample = loss_fn(y_pred, y)
            else:
                y = y.reshape(-1, 1)
                y_pred_metric = y_pred.cpu().numpy()
                val_loss_per_sample = loss_fn(y_pred, y)
                if val_loss_per_sample.ndim > 1:
                    val_loss_per_sample = val_loss_per_sample.mean(dim=1)

            y_true = y.cpu().numpy()

            # Calculate metric
            metric_val = get_metric(metric=self.metric, y_pred=y_pred_metric, y=y_true, use_fairness=self.use_fairness)

            if self.use_fairness:
                loss_fair = self.lambda_fair * group_dro_loss(loss_fair_fn, fair_val, val_loss_per_sample)
                metric_val = metric_val - loss_fair.item()
                
            return metric_val
   

    def predict(self, X):

        y_pred = self.layers_(X)
        return y_pred

class CNN_Model(NN_model):

    def __init__(self, mode_problem='classification', hidden_layers=[100], optimizer='sgd', batch_size=128, max_epoch=100, validation_fraction=0.1, early_stopping=0, address=None, test_batch_size=1000, random_seed=666, lr=0.001, normalize=True, device=None, strides=[2], kernel_sizes=[2], channels=[1], global_averaging=False, dropout=0, num_of_classes=2,use_fairness=False, fairness_attr=[0], num_of_fair=2, lambda_fair=1.0):

        super().__init__(mode_problem, True, hidden_layers, optimizer, batch_size, max_epoch, validation_fraction, early_stopping, address, test_batch_size, random_seed, lr, num_of_classes, normalize, device, use_fairness, fairness_attr, num_of_fair, lambda_fair)
        if strides is None:
            self.strides = [1] * len(kernel_sizes)
        else:
            self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        assert len(channels)==len(kernel_sizes), 'Invalid channels or kernel_sizes'
        self.global_averaging = global_averaging
        self.dropout = 0
        

    def build(self, X, y, bias=True, init_=True, data_dict=None):
        self.is_built = True
        self.input_dim = X.shape[1:]   # e.g. (3, 32, 32)
        if isinstance(y, np.ndarray):
            y = y.ravel().tolist()
            if self.num_classes is None:
                self.num_classes = len(set(y))
        else:
            if self.num_classes is None:
                
                self.num_classes = len(set(y))

        
        in_channels = 1 if len(self.input_dim) == 2 else self.input_dim[0]
        self.layers_ = nn.Sequential()
        for i, (k_size, out_ch, stride) in enumerate(zip(self.kernel_sizes, self.channels, self.strides)):
            conv_layer = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_ch,
                                kernel_size=k_size,
                                stride=stride,
                                device=self.device)
            if init_:
                nn.init.kaiming_uniform_(conv_layer.weight, a=np.sqrt(5))
            self.layers_.add_module(f'conv_{i}_CW', conv_layer)
            self.layers_.add_module(f'activation_{i}', nn.ReLU())
            in_channels = out_ch

        # compute flattened dim with a dummy input shaped by input_dim (C,H,W)
        with torch.no_grad():
            dummy_shape = (1,) + tuple(self.input_dim)
            dummy_input = torch.zeros(dummy_shape, device=self.device)
            conv_output = self.layers_(dummy_input)
            if self.global_averaging:
                flattened_dim = conv_output.shape[1]
            else:
                flattened_dim = int(np.prod(conv_output.shape[1:]))

        self.dense_net = nn.Sequential()
        current_dim = flattened_dim
        for i, hidden_layer in enumerate(self.hidden_layers):
            
            self.dense_net.add_module(f'linear_layer_{i}', nn.Linear(current_dim, hidden_layer, bias=False, dtype=torch.float, device=self.device))
            self.dense_net.add_module(f'Relu_{i}', nn.ReLU())
            current_dim = hidden_layer

        self.final_layer = nn.Linear(current_dim, 1 if self.problem == 'regression' else self.num_classes, dtype=torch.float, device=self.device)

    def predict(self, X):
        if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame):
            X = transform_to_torch(arr=X).to(self.device)

        # Reshape if input is flat (e.g., MNIST 784 -> 1x28x28)
        if X.dim() == 2:
            side = int(np.sqrt(X.shape[1]))
            X = X.view(-1, 1, side, side)
        
        X = self.layers_(X)
        
        if self.global_averaging:
            # Global average pooling
            X = X.mean(dim=[2, 3])
        else:
            X = X.view(X.size(0), -1) # Flatten
        
        prelogits = self.dense_net(X)
        logits = self.final_layer(prelogits)

        if self.problem == 'regression':
            return logits.squeeze(-1)
        return logits
    