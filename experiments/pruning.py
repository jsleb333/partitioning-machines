from sklearn.model_selection import KFold
from sklearn.metrics import zero_one_loss, accuracy_score
import numpy as np
from scipy.special import zeta
from copy import copy, deepcopy
from typing import Callable

from partitioning_machines import DecisionTree
from partitioning_machines import wedderburn_etherington
from partitioning_machines import breiman_alpha_pruning_objective, growth_function_upper_bound


def prune_with_score(decision_tree: DecisionTree,
                     score_fn: Callable[[DecisionTree, DecisionTree], float],
                     minimize: bool = True) -> float:
    """Prune a decision tree classifier using a score function to compare unpruned tree with pruned ones. This corresponds to Algorithm 3 in Appendix E of the paper of Leboeuf et al. (2020).

    Args:
        decision_tree (DecisionTree):
            The fully grown decision tree classifier trained on some dataset.
        score_fn (Callable[[DecisionTree, DecisionTree], float]):
            A scoring function that characterizes the performance of the pruned tree. The function will receive as input the a pruned temporary copy of the original tree as well as a reference to the original subtree that was pruned. The package implements two score functions: ErrorScore and BoundScore.
        minimize (bool, optional):
            Determines if the score function should be minimized or maximized. Defaults to True.

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


class ErrorScore:
    def __init__(self, dtc, X, y) -> None:
        self.dtc = deepcopy(dtc)
        self.X, self.y = X, y

    def __call__(self, pruned_tree, subtree) -> float:
        self.dtc.tree = pruned_tree
        return 1 - accuracy_score(y_true=self.y, y_pred=self.dtc.predict(self.X))


class BoundScore:
    def __init__(self,
                 n_features: int,
                 nominal_feat_dist: list[int],
                 ordinal_feat_dist: list[int],
                 bound: Callable,
                 table: dict = {},
                 loose_pfub: bool = True,
                 errors_logprob_prior: Callable = None,
                 complexity_logprob_prior: Callable = None,
                 delta: float = .05) -> None:

        self.n_features = n_features
        self.nominal_feat_dist = nominal_feat_dist
        self.ordinal_feat_dist = ordinal_feat_dist
        self.bound = bound
        self.table = table
        self.loose_pfub = loose_pfub
        self.delta = delta

        self.errors_logprob_prior = errors_logprob_prior
        if errors_logprob_prior is None:
            r = 1/2
            self.errors_logprob_prior = lambda n_errors: np.log(1-r) + n_errors*np.log(r)

        self.complexity_logprob_prior = complexity_logprob_prior
        if complexity_logprob_prior is None:
            s = 2
            self.complexity_logprob_prior = lambda complexity_idx: -np.log(zeta(s)) - s*np.log(complexity_idx) - np.log(float(wedderburn_etherington(complexity_idx)))

    def __call__(self, pruned_tree, subtree):
        n_classes = pruned_tree.n_examples_by_label.shape[0]
        growth_function = growth_function_upper_bound(pruned_tree,
                                                      self.n_features,
                                                      nominal_feat_dist=self.nominal_feat_dist,
                                                      ordinal_feat_dist=self.ordinal_feat_dist,
                                                      n_classes=n_classes,
                                                      pre_computed_tables=self.table,
                                                      loose=self.loose_pfub)
        n_examples = pruned_tree.n_examples
        n_errors = pruned_tree.n_errors
        errors_logprob = self.errors_logprob_prior(n_errors)
        complexity_logprob = self.complexity_logprob_prior(pruned_tree.n_leaves)

        return self.bound(n_examples, n_errors, growth_function, errors_logprob, complexity_logprob, self.delta)


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
