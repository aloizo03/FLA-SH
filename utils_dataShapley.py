from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from model import CNN_Model, NN_model
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

from CNN_Test_model import Cnn_Model_3Layers

def create_model(model_type, problem, max_epochs, batch_size, lr, val_frac, hidden_layer_size=(20,), use_img=False, use_NN=True, num_classes=10, norm=True):
    if model_type == 'logistic':
        model = LogisticRegression(n_jobs=2,
                                   max_iter=3000,
                                   random_state=666)
    elif model_type == 'Tree':
        model = DecisionTreeClassifier(random_state=666)
    elif model_type == 'MLP_Regression':
        if isinstance(hidden_layer_size, list):
            hidden_layer_size = list(hidden_layer_size)
        if use_NN:
            model =  NN_model(mode_problem=problem, batch_size=batch_size, max_epoch=max_epochs,
                                            lr=lr, validation_fraction=val_frac, optimizer='sgd', address=None, 
                                            hidden_layers=hidden_layer_size,
                                            num_of_classes=num_classes,
                                            normalize=norm)
        else:
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_size,
                             activation='relu', 
                             solver='sgd',
                             warm_start=False,
                             max_iter=5000,
                             early_stopping=False)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=666)
    elif model_type == 'MLP_Class':
        if isinstance(hidden_layer_size, list):
            hidden_layer_size = list(hidden_layer_size)
        if use_img:
            model = CNN_Model(mode_problem=problem, 
                                batch_size=batch_size, 
                                max_epoch=max_epochs, 
                                validation_fraction=val_frac, 
                                optimizer='sgd', 
                                address=None, 
                                hidden_layers=hidden_layer_size, 
                                early_stopping=0, 
                                normalize=norm,
                                num_of_classes=num_classes)
        else:
            if use_NN:
               model =  NN_model(mode_problem=problem, batch_size=batch_size, max_epoch=max_epochs,
                                            lr=lr, validation_fraction=val_frac, optimizer='sgd', address=None, 
                                            hidden_layers=hidden_layer_size,
                                            num_of_classes=num_classes,
                                            normalize=norm)
            else:
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_size,
                                    activation='relu',
                                    solver='sgd',
                                    learning_rate_init=lr,
                                    warm_start=False,
                                    max_iter=max_epochs,
                                    early_stopping=False, 
                                    learning_rate='adaptive')

    return model

def calculate_error(error_):
    if len(error_) <= 15:
        return 1.0
    
    all_vals = (np.cumsum(error_, 0)/np.reshape(np.arange(1, len(error_)+1), (-1,1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)


def scale_to_grid(vectors, bits):
    max_val = np.max(vectors, axis=0)
    min_val = np.min(vectors, axis=0)
    scaled = (vectors - min_val) / (max_val - min_val + 1e-9)
    scaled *= (2 ** bits - 1)
    return np.floor(scaled).astype(int)

def merge_small_chunks(data_tuples_embds, chunks_idx, chunk_size):

    flag = True
    while flag:
        flag = False
        centroids = {c_idx: data_tuples_embds[idxs].mean(axis=0) for c_idx, idxs in chunks_idx.items()}
        small_chunks = sorted(chunks_idx.items(), key=lambda kv: len(kv[1]))

        for c_idx, idxs in small_chunks:
            if len(idxs) <= 2:
                rest_centroids = [c for c in chunks_idx.keys() if c != c_idx]

                other_centroids = [centroids[c] for c in rest_centroids]

                cid_vec = centroids[c_idx].reshape(1, -1)
                nearest_idx, _ = pairwise_distances_argmin_min(cid_vec, other_centroids)
                target_centroid_idx = rest_centroids[nearest_idx[0]]
                if len(chunks_idx[target_centroid_idx]) == chunk_size or (len(chunks_idx[target_centroid_idx])+len(idxs)) > chunk_size:
                    for idx in nearest_idx[1:]:
                        target_centroid_idx = rest_centroids[nearest_idx[0]]
                        if len(chunks_idx[target_centroid_idx]) == chunk_size or (len(chunks_idx[target_centroid_idx])+len(idxs)) > chunk_size:
                            break
                chunks_idx[target_centroid_idx] = np.concatenate([chunks_idx[target_centroid_idx], chunks_idx[c_idx]])
                del chunks_idx[c_idx]
                flag = False
    return chunks_idx

    
                        

