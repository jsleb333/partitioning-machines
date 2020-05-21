from sklearn.model_selection import KFold
from sklearn.metrics import zero_one_loss
import numpy as np
from copy import copy

from partitioning_machines import breiman_alpha_pruning_objective


def prune_with_bound(decision_tree, bound):
    
    leaf = decision_tree.tree
    while not leaf.is_leaf():
        leaf = leaf.left_subtree
    best_bound = bound(leaf)
    bounds_value = decision_tree.compute_pruning_coefficients(bound)
    
    while bounds_value[0] <= best_bound:
        best_bound = bounds_value[0]
        decision_tree.prune_tree(best_bound)
        bounds_value = decision_tree.compute_pruning_coefficients(bound)
    
    return best_bound


def prune_with_cv(
        decision_tree,
        X,
        y,
        n_folds=10,
        pruning_objective=breiman_alpha_pruning_objective,
        optimisation_mode='min'):
    pruning_coefs = decision_tree.compute_pruning_coefficients(pruning_objective)
    
    CV_trees = [copy(decision_tree) for i in range(n_folds)]
    
    fold_idx = list(KFold(n_splits=n_folds).split(X))
    
    for fold, (tr_idx, ts_idx) in enumerate(fold_idx):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        CV_trees[fold].fit(X_tr, y_tr)
    
    n_errors = [0] * len(pruning_coefs)
    for k, threshold in enumerate(pruning_coefs):
        for tree, (tr_idx, ts_idx) in zip(CV_trees, fold_idx):
            tree.prune_tree(threshold, pruning_objective)
            X_ts, y_ts = X[ts_idx], y[ts_idx]
            y_pred = tree.predict(X_ts)
            n_errors[k] += zero_one_loss(y_true=y_ts, y_pred=y_pred, normalize=False)
    
    argext = np.argmin if optimisation_mode == 'min' else np.argmax
    optimal_pruning_coef_threshold = pruning_coefs[argext(n_errors)]
    decision_tree.prune_tree(optimal_pruning_coef_threshold)
    
    return optimal_pruning_coef_threshold
