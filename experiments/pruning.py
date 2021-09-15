from re import sub
from sklearn.model_selection import KFold
from sklearn.metrics import zero_one_loss
import numpy as np
from copy import copy
from typing import Callable

from partitioning_machines import breiman_alpha_pruning_objective, DecisionTree


def prune_with_score(decision_tree: DecisionTree,
                     score_fn: Callable[[DecisionTree, DecisionTree], float],
                     minimize: bool = True) -> float:
    """Prune a decision tree classifier using a score function to compare unpruned tree with pruned ones. This corresponds to Algorithm 3 in Appendix E of the paper of Leboeuf et al. (2020).

    Args:
        decision_tree (DecisionTree): The fully grown decision tree classifier trained on some dataset.
        score_fn (Callable[[DecisionTree, DecisionTree], float]): A scoring function that characterizes the performance of the pruned tree. The function will receive as input the a pruned temporary copy of the original tree as well as a reference to the original subtree that was pruned.
        minimize (bool, optional): Determines if the score function should be minimized or maximized. Defaults to True.

    Returns the best score (a float).
    """
    sign = 1 if minimize else -1

    best_score = tmp_best_score = score_fn(decision_tree, decision_tree)

    while not decision_tree.is_leaf():
        new_best_found = False
        for subtree in decision_tree:
            if subtree.is_leaf():
                continue
            tmp_pruned_tree = subtree.remove_subtree(inplace=False).root
            tmp_score = score_fn(tmp_pruned_tree, subtree)
            if tmp_score*sign <= tmp_best_score*sign:
                tmp_best_score = tmp_score
                tmp_best_subtree = subtree
                new_best_found = True

        if new_best_found:
            tmp_best_subtree.remove_subtree(inplace=True) # Prunes the original decision_tree
            best_score = tmp_best_score
        else:
            break

    return best_score


def prune_with_cv(
        decision_tree,
        X,
        y,
        n_folds=10,
        pruning_objective=breiman_alpha_pruning_objective,
        optimisation_mode='min'):
    """
    Pruning using cross-validation. This is an abstraction of CART's cost-complexity pruning algorithm, where the objective can be whatever we want. The optimal pruning threshold is chosen as described by Breiman (1984).
    """
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

    sign = 1 if optimisation_mode == 'min' else -1

    argmin_first = 0
    argmin_last = 0
    val_min = np.infty
    for i, errors in enumerate(n_errors):
        if sign*errors < sign*val_min:
            argmin_first = i
            argmin_last = i
            val_min = errors
        elif errors == val_min:
            argmin_last = i

    optimal_pruning_coef_threshold = (pruning_coefs[argmin_first] + pruning_coefs[argmin_last])/2
    decision_tree.prune_tree(optimal_pruning_coef_threshold)

    return optimal_pruning_coef_threshold
