from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import zero_one_loss, accuracy_score
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from copy import copy

from partitioning_machines import DecisionTreeClassifier, breiman_alpha_pruning_objective, gini_impurity_criterion, decision_tree_to_tikz


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
    decision_tree = DecisionTreeClassifier(gini_impurity_criterion)
    X, y = load_iris(return_X_y=True)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.25, random_state=42)
    n_examples, n_features = X.shape
    
    print(
        f'''Loading IRIS dataset. Stats:
                n_examples (total): {n_examples},
                n_examples (train): {X_tr.shape[0]},
                n_examples (test): {X_ts.shape[0]},
                n_features: {n_features}''')
    
    decision_tree.fit(X_tr, y_tr)
    acc_tr = accuracy_score(y_true=y_tr, y_pred=decision_tree.predict(X_tr))
    acc_ts = accuracy_score(y_true=y_ts, y_pred=decision_tree.predict(X_ts))
    print(f'Accuracy score of original tree on train dataset: {acc_tr:.3f}')
    print(f'Accuracy score of original tree on test dataset: {acc_ts:.3f}')

    original_tree = decision_tree_to_tikz(decision_tree, decision_tree.label_encoder.labels)
    optimal_threshold = cross_validate_pruning_coef_threshold(decision_tree, X_tr, y_tr, n_fold=5)
    print(f'Optimal cross-validated pruning coefficient threshold: {optimal_threshold:.3f}')

    pruned_tree = decision_tree_to_tikz(decision_tree, decision_tree.label_encoder.labels)
    acc_tr = accuracy_score(y_true=y_tr, y_pred=decision_tree.predict(X_tr))
    acc_ts = accuracy_score(y_true=y_ts, y_pred=decision_tree.predict(X_ts))
    print(f'Accuracy score of pruned tree on train dataset: {acc_tr:.3f}')
    print(f'Accuracy score of pruned tree on test dataset: {acc_ts:.3f}')
    
    import python2latex as p2l

    doc = p2l.Document('breinamn_tree_pruning', path='./experiments/', doc_type='standalone', border='1cm')
    doc.add_package('tikz')
    del doc.packages['geometry']
    doc.add_to_preamble('\\usetikzlibrary{shapes}')
    doc += original_tree
    doc += r'\hspace{1cm}'
    doc += pruned_tree
    doc.build()