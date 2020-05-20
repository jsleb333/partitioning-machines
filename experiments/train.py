from sklearn import datasets as dataset_loader
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss, accuracy_score
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from copy import copy, deepcopy

from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion
from partitioning_machines import breiman_alpha_pruning_objective
from partitioning_machines import vapnik_bound_pruning_objective_factory
from pruning import *


def train(X, y, n_folds=10):
    decision_tree = DecisionTreeClassifier(gini_impurity_criterion)
    n_examples, n_features = X.shape
    bound = vapnik_bound_pruning_objective_factory(n_features)
    
    decision_tree.fit(X, y)
    
    pruned_with_bound_tree = deepcopy(decision_tree)
    prune_with_bound(pruned_with_bound_tree, bound)

    pruned_with_cv_tree = deepcopy(decision_tree)
    prune_with_cv(pruned_with_cv_tree, X, y, n_folds=n_folds)
    
    return decision_tree, pruned_with_bound_tree, pruned_with_cv_tree


if __name__ == "__main__":
    
    test_split_ratio = .25
    n_draws = 10
    n_folds = 10
    
    datasets = {'iris':dataset_loader.load_iris(),
                # 'digits':dataset_loader.load_digits(),
                'wine':dataset_loader.load_wine(),
                'breast_cancer':dataset_loader.load_breast_cancer(),
    }
    
    for name, dataset in datasets.items():
        X = dataset.data
        y = dataset.target
        n_examples = X.shape[0]
        X.reshape((n_examples, -1))
        n_features = X.shape[1]
        
        acc_tr = [np.zeros(n_draws) for _ in range(3)]
        acc_ts = [np.zeros(n_draws) for _ in range(3)]
        leaves = [np.zeros(n_draws) for _ in range(3)]
        
        for draw in range(n_draws):
            X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=test_split_ratio, random_state=draw*100)
            trees = train(X_tr, y_tr, n_folds)
            for i, tree in enumerate(trees):
                acc_tr[i][draw] = accuracy_score(y_tr, tree.predict(X_tr))
                acc_ts[i][draw] = accuracy_score(y_ts, tree.predict(X_ts))
                leaves[i][draw] = tree.tree.n_leaves
        
        acc_tr_mean = [acc.mean() for acc in acc_tr]
        acc_tr_std = [acc.std() for acc in acc_tr]
        
        acc_ts_mean = [acc.mean() for acc in acc_ts]
        acc_ts_std = [acc.std() for acc in acc_ts]
        
        leaves_mean = [l.mean() for l in leaves]
        leaves_std = [l.std() for l in leaves]
        
        print(f"""
Dataset: {name}
    Mean train accuracy:
        Original tree: {acc_tr_mean[0]} ± {acc_tr_std[0]}
        Vapnik tree: {acc_tr_mean[1]} ± {acc_tr_std[1]}
        Breiman tree: {acc_tr_mean[2]} ± {acc_tr_std[2]}
    Mean test accuracy:
        Original tree: {acc_ts_mean[0]} ± {acc_ts_std[0]}
        Vapnik tree: {acc_ts_mean[1]} ± {acc_ts_std[1]}
        Breiman tree: {acc_ts_mean[2]} ± {acc_ts_std[2]}
    Mean number of leaves:
        Original tree: {leaves_mean[0]} ± {leaves_std[0]}
        Vapnik tree: {leaves_mean[1]} ± {leaves_std[1]}
        Breiman tree: {leaves_mean[2]} ± {leaves_std[2]}
              """)
