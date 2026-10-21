from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from model import (CNN_Model, NN_model, LSTM_Model, LogisticRegression_Model,
                   ResNet_Model, SoftDecisionTree_Model, BoostedTree_Model,
                   NODE_Model, AlexNet_Model)
import numpy as np
import os
from sklearn.metrics import pairwise_distances_argmin_min
import math
from CNN_Test_model import Cnn_Model_3Layers

def create_model(model_type, problem, max_epochs, batch_size, lr, val_frac, hidden_layer_size=(20,),
                 use_img=False, use_NN=True, num_classes=10, norm=True, use_fairness=False, fairness_attr=0,
                 num_of_fair=2, lambda_fair=1.0, fair_metric='groupdro', compute_shapley=True,
                 early_stopping=None,
                 tree_depth=4, boosted_num_trees=10, boosted_tree_depth=3, boosted_shrinkage=None,
                 node_num_layers=2, node_trees_per_layer=8, node_depth=4):
    if early_stopping is None:
        _es_env = os.environ.get('FLASH_EARLY_STOPPING', '').strip()
        if _es_env:
            try:
                early_stopping = int(_es_env)
            except ValueError:
                print(f"[create_model] ignoring non-integer "
                      f"FLASH_EARLY_STOPPING={_es_env!r}")

    _es_img = 0 if early_stopping is None else early_stopping
    _es_kw = {} if early_stopping is None else {'early_stopping': early_stopping}

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
            model =  NN_model(**_es_kw, mode_problem=problem, batch_size=batch_size, max_epoch=max_epochs,
                                            lr=lr, validation_fraction=val_frac, optimizer='sgd', address=None, 
                                            hidden_layers=hidden_layer_size,
                                            num_of_classes=num_classes,
                                            normalize=norm,
                                            use_fairness=use_fairness,
                                            fairness_attr=fairness_attr, 
                                            num_of_fair=num_of_fair,
                                            lambda_fair=lambda_fair,
                                            fair_metric=fair_metric,
                                            compute_Shapley=compute_shapley)
        else:
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_size,
                             activation='relu', 
                             solver='sgd',
                             warm_start=False,
                             max_iter=5000,
                             early_stopping=False)
    elif model_type == 'logistic_sgd':
        # Torch multinomial logistic regression trained with SGD (tabular).
        model = LogisticRegression_Model(**_es_kw, mode_problem=problem, batch_size=batch_size,
                                         max_epoch=max_epochs, lr=lr,
                                         validation_fraction=val_frac, optimizer='sgd',
                                         address=None, num_of_classes=num_classes,
                                         normalize=norm, use_fairness=use_fairness,
                                         fairness_attr=fairness_attr,
                                         num_of_fair=num_of_fair, lambda_fair=lambda_fair,
                                         fair_metric=fair_metric,
                                         compute_Shapley=compute_shapley)
    elif model_type == 'soft_tree_sgd':
        # Differentiable ("gradient") decision tree trained with SGD (tabular).
        model = SoftDecisionTree_Model(**_es_kw, mode_problem=problem, batch_size=batch_size,
                                       max_epoch=max_epochs, lr=lr,
                                       validation_fraction=val_frac, optimizer='sgd',
                                       address=None, num_of_classes=num_classes,
                                       normalize=norm, use_fairness=use_fairness,
                                       fairness_attr=fairness_attr,
                                       num_of_fair=num_of_fair, lambda_fair=lambda_fair,
                                       fair_metric=fair_metric,
                                       compute_Shapley=compute_shapley,
                                       depth=tree_depth)
    elif model_type == 'boosted_tree_sgd':
        # Additive ensemble of soft decision trees ("boosted tree"), all
        # trained jointly with SGD (tabular).
        model = BoostedTree_Model(**_es_kw, mode_problem=problem, batch_size=batch_size,
                                  max_epoch=max_epochs, lr=lr,
                                  validation_fraction=val_frac, optimizer='sgd',
                                  address=None, num_of_classes=num_classes,
                                  normalize=norm, use_fairness=use_fairness,
                                  fairness_attr=fairness_attr,
                                  num_of_fair=num_of_fair, lambda_fair=lambda_fair,
                                  fair_metric=fair_metric,
                                  compute_Shapley=compute_shapley,
                                  num_trees=boosted_num_trees,
                                  tree_depth=boosted_tree_depth,
                                  shrinkage=boosted_shrinkage)
    elif model_type == 'node_sgd':
        model = NODE_Model(**_es_kw, mode_problem=problem, batch_size=batch_size,
                           max_epoch=max_epochs, lr=lr,
                           validation_fraction=val_frac, optimizer='adam',
                           address=None, num_of_classes=num_classes,
                           normalize=norm, use_fairness=use_fairness,
                           fairness_attr=fairness_attr,
                           num_of_fair=num_of_fair, lambda_fair=lambda_fair,
                           fair_metric=fair_metric,
                           compute_Shapley=compute_shapley,
                           num_layers=node_num_layers,
                           trees_per_layer=node_trees_per_layer,
                           depth=node_depth)
    elif model_type == 'ResNet':
        # ResNet trained with SGD for image datasets.
        if isinstance(hidden_layer_size, tuple):
            hidden_layer_size = list(hidden_layer_size)
        model = ResNet_Model(mode_problem=problem, batch_size=batch_size,
                             max_epoch=max_epochs, lr=lr,
                             validation_fraction=val_frac, optimizer='sgd',
                             address=None, hidden_layers=hidden_layer_size,
                             early_stopping=_es_img, normalize=norm,
                             num_of_classes=num_classes, use_fairness=use_fairness,
                             fairness_attr=fairness_attr, num_of_fair=num_of_fair,
                             lambda_fair=lambda_fair, fair_metric=fair_metric,
                             compute_Shapley=compute_shapley)
    elif model_type == 'AlexNet':
        # AlexNet-style CNN trained with SGD for image datasets (see
        # AlexNet_Model's docstring for exactly how it's adapted from the
        # original architecture for small 28x28/32x32 inputs).
        if isinstance(hidden_layer_size, tuple):
            hidden_layer_size = list(hidden_layer_size)
        model = AlexNet_Model(mode_problem=problem, batch_size=batch_size,
                              max_epoch=max_epochs, lr=lr,
                              validation_fraction=val_frac, optimizer='sgd',
                              address=None, hidden_layers=hidden_layer_size,
                              early_stopping=_es_img, normalize=norm,
                              num_of_classes=num_classes, use_fairness=use_fairness,
                              fairness_attr=fairness_attr, num_of_fair=num_of_fair,
                              lambda_fair=lambda_fair, fair_metric=fair_metric,
                              compute_Shapley=compute_shapley)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=666)
    elif model_type == 'LSTM':
            model = LSTM_Model(**_es_kw, mode_problem=problem, batch_size=batch_size, max_epoch=max_epochs, 
                                lr=lr, validation_fraction=val_frac, optimizer='sgd', address=None, 
                                hidden_layers=hidden_layer_size,
                                num_of_classes=num_classes,
                                normalize=norm,
                                use_fairness=use_fairness,
                                fairness_attr=fairness_attr, 
                                num_of_fair=num_of_fair,
                                lambda_fair=lambda_fair,
                                fair_metric=fair_metric,
                                compute_Shapley=compute_shapley)
    elif model_type == 'MLP_Class':
        if isinstance(hidden_layer_size, list):
            hidden_layer_size = list(hidden_layer_size)
        if use_img:
            model = CNN_Model(mode_problem=problem,
                                batch_size=batch_size,
                                max_epoch=max_epochs,
                                lr=lr,
                                validation_fraction=val_frac,
                                optimizer='sgd',
                                address=None,
                                hidden_layers=hidden_layer_size,
                                early_stopping=_es_img,
                                normalize=norm,
                                num_of_classes=num_classes,
                                use_fairness=use_fairness,
                                fairness_attr=fairness_attr,
                                num_of_fair=num_of_fair,
                                lambda_fair=lambda_fair,
                                fair_metric=fair_metric,
                                compute_Shapley=compute_shapley)
        else:
            if use_NN:
               model =  NN_model(**_es_kw, mode_problem=problem, batch_size=batch_size, max_epoch=max_epochs,
                                            lr=lr, validation_fraction=val_frac, optimizer='sgd', address=None, 
                                            hidden_layers=hidden_layer_size,
                                            num_of_classes=num_classes,
                                            normalize=norm,
                                            use_fairness=use_fairness,
                                            fairness_attr=fairness_attr, 
                                            num_of_fair=num_of_fair, 
                                            lambda_fair=lambda_fair,
                                            fair_metric=fair_metric,
                                            compute_Shapley=compute_shapley)
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


def _shapley_log_p(s, c):
    """log p(Z) for |Z|=s out of c total chunks: log(s!(c-s-1)!/c!)."""
    if s < 0 or s > c - 1:
        return float('-inf')
    return math.lgamma(s + 1) + math.lgamma(c - s) - math.lgamma(c + 1)


def _chunk_inclusion_probs(c, base_incl=0.5, strength=0.5, pi_floor=0.02, pi_ceil=0.98):
    """Per-chunk inclusion probabilities pi_i, from RANK alone (position 0 = highest priority in the already priority-sorted chunk order -- SR score,
    Hilbert index, or fairness purity, whichever permutation ran). This uses only information available before any coalition is sampled; it never
    looks at v(Z) for any multi-chunk coalition.

    strength=0 -> uniform (pi_i = base_incl for every i, no priority tilt).
    strength=1 -> maximally tilted between pi_floor and pi_ceil by rank.
    """
    if c <= 1:
        return np.full(max(c, 1), base_incl)
    rank = np.arange(c)                                    # 0 = best (highest priority)
    rank_norm = 1.0 - rank / (c - 1)                        # 1 = best, 0 = worst
    pi = base_incl + strength * (rank_norm - 0.5) * (pi_ceil - pi_floor)
    return np.clip(pi, pi_floor, pi_ceil)


def _log_bernoulli_q(included, pi):
    """Exact log q(Z) = sum_i [included_i*log(pi_i) + (1-included_i)*log(1-pi_i)]."""
    included = np.asarray(included, dtype=bool)
    pi = np.asarray(pi, dtype=float)
    return float(np.sum(np.where(included, np.log(pi), np.log1p(-pi))))


def _sample_bernoulli_coalition(pi, rng):
    """Draw Z via independent per-chunk inclusion with probability pi_i."""
    draws = rng.random(len(pi)) < pi
    return set(np.flatnonzero(draws).tolist()), draws


def _sample_uniform_size_coalition(c, rng):
    """Draw Z ~ p directly: size ~ Uniform{0,...,c-1}, then a uniform subset
    of that size -- the exact marginal of a random-permutation prefix."""
    s = int(rng.integers(0, c))            # 0..c-1 inclusive
    idxs = rng.choice(c, size=s, replace=False)
    included = np.zeros(c, dtype=bool)
    included[idxs] = True
    return set(idxs.tolist()), included


def _sample_coalition_qtilde(c, pi, epsilon, rng):
    """Draw one coalition from qtilde = (1-eps)*q + eps*p and return
    (Z, importance_weight) with importance_weight = p(Z) / qtilde(Z).

    qtilde(Z) is evaluated as the closed-form sum of BOTH component
    densities regardless of which branch actually produced the draw (that is
    what makes mixture importance sampling exact), so the returned weight is
    exact -- not approximated -- for every draw.
    """
    if rng.random() < epsilon:
        Z, included = _sample_uniform_size_coalition(c, rng)
    else:
        Z, included = _sample_bernoulli_coalition(pi, rng)

    s = int(included.sum())
    log_p = _shapley_log_p(s, c)
    log_q = _log_bernoulli_q(included, pi)

    log_eps = math.log(epsilon) if epsilon > 0 else float('-inf')
    log_1meps = math.log1p(-epsilon) if epsilon < 1 else float('-inf')
    a = log_1meps + log_q
    b = log_eps + log_p
    m = max(a, b)
    log_qtilde = m + math.log(math.exp(a - m) + math.exp(b - m))

    return Z, math.exp(log_p - log_qtilde)
                        

