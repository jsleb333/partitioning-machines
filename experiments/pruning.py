from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import zero_one_loss, accuracy_score
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from copy import copy, deepcopy

from graal_utils import Timer

from partitioning_machines import DecisionTreeClassifier, breiman_alpha_pruning_objective, gini_impurity_criterion, decision_tree_to_tikz
from partitioning_machines import shawe_taylor_bound_pruning_objective_factory, \
                                  vapnik_bound_pruning_objective_factory


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
        print(bounds_value)
    
    return best_bound


def cross_validate_pruning_coef_threshold(
        decision_tree,
        X,
        y,
        n_fold=10,
        pruning_objective=breiman_alpha_pruning_objective,
        optimisation_mode='min'):
    pruning_coefs = decision_tree.compute_pruning_coefficients(pruning_objective)
    
    CV_trees = [copy(decision_tree) for i in range(n_fold)]
    
    fold_idx = KFold(n_splits=n_fold).split(X)
    
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


if __name__ == '__main__':
    
    with Timer():    
        decision_tree = DecisionTreeClassifier(gini_impurity_criterion)
        X, y = load_iris(return_X_y=True)
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.25, random_state=40)
        n_examples, n_features = X.shape
        classes = ['Setosa', 'Versicolour', 'Virginica']
        table = {}
        bound = vapnik_bound_pruning_objective_factory(n_features, table)
        # bound = shawe_taylor_bound_pruning_objective_factory(n_features, table)
        
        print(
            f'''Loading IRIS dataset. Stats:
                    n_examples (total): {n_examples},
                    n_examples (train): {X_tr.shape[0]},
                    n_examples (test): {X_ts.shape[0]},
                    n_features: {n_features}''')
        
        decision_tree.fit(X_tr, y_tr)
        acc_tr_orig = accuracy_score(y_true=y_tr, y_pred=decision_tree.predict(X_tr))
        acc_ts_orig = accuracy_score(y_true=y_ts, y_pred=decision_tree.predict(X_ts))
        print(f'Accuracy score of original tree on train dataset: {acc_tr_orig:.3f}')
        print(f'Accuracy score of original tree on test dataset: {acc_ts_orig:.3f}')

        original_tree = decision_tree_to_tikz(decision_tree, classes)
        copy_of_tree = deepcopy(decision_tree)
        best_bound = prune_with_bound(decision_tree, bound)
        print(f'Best bound: {best_bound:.3f}')

        pruned_tree_with_bound = decision_tree_to_tikz(decision_tree, classes)
        acc_tr_bound = accuracy_score(y_true=y_tr, y_pred=decision_tree.predict(X_tr))
        acc_ts_bound = accuracy_score(y_true=y_ts, y_pred=decision_tree.predict(X_ts))
        print(f'Accuracy score of pruned tree on train dataset: {acc_tr_bound:.3f}')
        print(f'Accuracy score of pruned tree on test dataset: {acc_ts_bound:.3f}')

        decision_tree = copy_of_tree
        n_fold = 10
        optimal_threshold = cross_validate_pruning_coef_threshold(decision_tree, X_tr, y_tr, n_fold=n_fold)
        print(f'Optimal cross-validated pruning coefficient threshold: {optimal_threshold:.3f}')
        pruned_tree_with_cv = decision_tree_to_tikz(decision_tree, classes)
        acc_tr_cv = accuracy_score(y_true=y_tr, y_pred=decision_tree.predict(X_tr))
        acc_ts_cv = accuracy_score(y_true=y_ts, y_pred=decision_tree.predict(X_ts))
        print(f'Accuracy score of pruned tree on train dataset: {acc_tr_cv:.3f}')
        print(f'Accuracy score of pruned tree on test dataset: {acc_ts_cv:.3f}')
        
        import python2latex as p2l

        doc = p2l.Document('tree_pruning_comparison', path='./experiments/', doc_type='standalone', border='1cm')
        doc.add_package('tikz')
        del doc.packages['geometry']
        doc.add_to_preamble('\\usetikzlibrary{shapes}')
        
        table = doc.new(p2l.Table((5,1), as_float_env=False))
        table[0,0] = 'Iris dataset'
        table[0,0].add_rule()
        table[1,0] = f'Number of examples (total): {n_examples}'
        table[2,0] = f'Train-test split: {X_tr.shape[0]}:{X_ts.shape[0]}'
        table[3,0] = f'Number of features: {n_features}'
        table[4,0] = f'Number of fold in CV: {n_fold}'
        
        table = doc.new(p2l.Table((3,3), as_float_env=False, bottom_rule=False, top_rule=False))
        table[0,0] = 'Full tree (no pruning)'
        original_tree.kwoptions['baseline'] = '(current bounding box.north)'
        table[1,0] = original_tree
        subtable = table[2,0].divide_cell((2,1))
        subtable[0,0] = f'Train acc: {acc_tr_orig:.3f}'
        subtable[1,0] = f'Test acc: {acc_ts_orig:.3f}'
        
        table[0,1] = "Vapnik's bound pruning"
        pruned_tree_with_bound.kwoptions['baseline'] = '(current bounding box.north)'
        table[1,1] = pruned_tree_with_bound
        subtable = table[2,1].divide_cell((3,1))
        subtable[0,0] = f'Train acc: {acc_tr_bound:.3f}'
        subtable[1,0] = f'Test acc: {acc_ts_bound:.3f}'
        subtable[2,0] = f'Bound value: {best_bound:.3f}'

        table[0,2] = "Breiman pruning with CV"
        pruned_tree_with_cv.kwoptions['baseline'] = '(current bounding box.north)'
        table[1,2] = pruned_tree_with_cv
        subtable = table[2,2].divide_cell((2,1))
        subtable[0,0] = f'Train acc: {acc_tr_cv:.3f}'
        subtable[1,0] = f'Test acc: {acc_ts_cv:.3f}'
        
        doc.build()