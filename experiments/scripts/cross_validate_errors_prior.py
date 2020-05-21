from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import zero_one_loss, accuracy_score
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from copy import copy, deepcopy

from graal_utils import Timer

from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion
from partitioning_machines import breiman_alpha_pruning_objective, modified_breiman_pruning_objective_factory
from partitioning_machines import vapnik_bound_pruning_objective_factory
from experiments.pruning import prune_with_bound


with Timer():
    X, y = load_iris(return_X_y=True)
    n_examples, n_features = X.shape
    dtc = DecisionTreeClassifier(gini_impurity_criterion)
    n_folds = 5

    radii = [2**-i for i in range(1, 16+1, 3)]

    cv_dtc = [copy(dtc) for i in range(n_folds)]

    fold_idx = list(KFold(n_splits=n_folds, shuffle=True, random_state=42).split(X))

    for fold, (tr_idx, ts_idx) in enumerate(fold_idx):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_ts, y_ts = X[ts_idx], y[ts_idx]
        cv_dtc[fold].fit(X_tr, y_tr)
        print(f'Fold {fold}, train acc', accuracy_score(y_tr, cv_dtc[fold].predict(X_tr)), 'test acc', accuracy_score(y_ts, cv_dtc[fold].predict(X_ts)))
        print(cv_dtc[fold].tree.n_leaves)

    n_errors = [0] * len(radii)
    for k, r in enumerate(radii):
        bound = vapnik_bound_pruning_objective_factory(
                                n_features,
                                errors_prob_prior=lambda n_err: (1-r) * r**n_err)
        for tree, (fold, (tr_idx, ts_idx)) in zip(cv_dtc, enumerate(fold_idx)):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_ts, y_ts = X[ts_idx], y[ts_idx]
            
            copy_of_tree = deepcopy(tree)
            prune_with_bound(copy_of_tree, bound)
            
            # print(f'Fold {fold}, train acc', accuracy_score(y_tr, copy_of_tree.predict(X_tr)), 'test acc', accuracy_score(y_ts, copy_of_tree.predict(X_ts)))
            print(copy_of_tree.tree.n_leaves)
            
            y_pred = copy_of_tree.predict(X_ts)
            n_errors[k] += zero_one_loss(y_true=y_ts, y_pred=y_pred, normalize=False)
        
        print(f'Radius {r}:\t mean test accuracy: {1 - n_errors[k]/n_examples}.')

    optimal_radii = radii[np.argmin(n_errors)]
    print(f'\nBest radius: {optimal_radii}.')
