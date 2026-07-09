import pandas as pd
import torch
import numpy as np
from sklearn.metrics import r2_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, roc_auc_score, root_mean_squared_error
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class GroupDRO:
    def __init__(self, n_groups, eta=0.1, device='cpu'):
        self.n_groups = n_groups
        self.eta = eta
        self.device = device
        self.q = torch.ones(n_groups, device=device) / n_groups

    @torch.no_grad()
    def update(self, group_losses):
        self.q *= torch.exp(self.eta * group_losses)
        self.q /= self.q.sum()

    def loss(self, group_losses):
        q = self.q.detach()
        return torch.sum(q * group_losses)


def group_dro_loss(group_dro: GroupDRO, groups, per_sample_losses):
    device = group_dro.device

    min_len = min(len(groups), len(per_sample_losses))
    if len(groups) != len(per_sample_losses):
        groups = groups[:min_len]
        per_sample_losses = per_sample_losses[:min_len]

    group_losses = torch.zeros(group_dro.n_groups, device=device)

    for g in range(group_dro.n_groups):
        mask = (groups == g)
        if mask.any():
            group_losses[g] = per_sample_losses[mask].mean()
        else:
            group_losses[g] = 0.0

    group_dro.update(group_losses.detach())
    return group_dro.loss(group_losses)


def variance_group_loss(per_sample_losses, groups, lambda_var=1.0):
    group_losses = []

    for g in torch.unique(groups):
        mask = (groups == g)
        group_losses.append(per_sample_losses[mask].mean())

    group_losses = torch.stack(group_losses)

    return group_losses.mean() + lambda_var * group_losses.var()


def cvar_loss(per_sample_losses, alpha=0.3):
    N = per_sample_losses.shape[0]

    if N == 0:
        return per_sample_losses.sum() * 0.0  # differentiable zero

    if alpha <= 0.0:
        return per_sample_losses.mean()

    # Clamp k to [1, N] so topk never receives k=0
    k = max(1, min(N, int(np.ceil(alpha * N))))

    # topk returns the k LARGEST values — correct for worst-case CVaR
    topk_losses, _ = torch.topk(per_sample_losses, k)
    return topk_losses.mean()


def demographic_parity_loss_multi(logits, protected_attr):
    """
    logits: tensor of shape (batch,)
    protected_attr: tensor of shape (batch,) with non binary values {0,1, ..., n} where n=number of values
    """
    probs = torch.sigmoid(logits)
    groups = protected_attr.unique()
    group_means = [probs[protected_attr == g].mean() for g in groups]
    mean_overall = torch.stack(group_means).mean()
    return torch.mean((torch.stack(group_means) - mean_overall) ** 2)


def demographic_parity_loss(logits, protected_attr):
    """
    logits: tensor of shape (batch,)
    protected_attr: tensor of shape (batch,) with binary values {0,1}
    """
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)

    # Masks for each protected group
    mask0 = (protected_attr == 0)
    mask1 = (protected_attr == 1)

    # Handle cases where a group might not appear in the batch
    if mask0.sum() == 0 or mask1.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    # Mean predicted probability for each group
    mean0 = probs[mask0].mean()
    mean1 = probs[mask1].mean()

    # Squared difference (demographic parity loss)
    dp_loss = (mean0 - mean1).pow(2)
    return dp_loss


def transform_to_torch(arr):
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        arr = arr.to_numpy()
    if isinstance(arr, list):
        arr = np.array(arr)

    tens = torch.tensor(arr, dtype=torch.float)
    if tens.dim() == 5 and tens.size(0) == 1:
        tens = tens.squeeze(0)
    return tens


def get_metric(metric, y_pred, y, clf=None, X=None, use_fairness=False):
    if metric.lower() == 'accuracy':
        return accuracy_score(y, y_pred)
    elif metric.lower() == 'f1-score':
        if len(set(y)) > 2:
            return f1_score(y_true=y, y_pred=y_pred, average='macro')
        else:
            return f1_score(y_true=y, y_pred=y_pred, average='binary')
    elif metric.lower() == 'auc':
        if len(set(y)) > 2:
            return roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')
        else:
            return roc_auc_score(y, clf.predict_proba(X)[:, 1])
    elif metric.lower() == 'mse':
        return mean_squared_error(y, y_pred)
    elif metric.lower() == 'rmse':
        return root_mean_squared_error(y, y_pred)


def _safe_split(X, y, val_fraction, fair_attrbs=None):
    """
    Split X, y (and optionally fair_attrbs) into train/val.
    Returns (X_tr, X_v, y_tr, y_v, f_tr, f_v).
    If too few samples for the split, returns val as None.
    """
    n = X.shape[0] if hasattr(X, 'shape') else len(X)
    min_needed = max(2, int(1 / val_fraction) + 1)
    if n < min_needed:
        return X, None, y, None, fair_attrbs, None
    if fair_attrbs is not None:
        X_tr, X_v, y_tr, y_v, f_tr, f_v = train_test_split(
            X, y, fair_attrbs, test_size=val_fraction)
        return X_tr, X_v, y_tr, y_v, f_tr, f_v
    else:
        X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=val_fraction)
        return X_tr, X_v, y_tr, y_v, None, None