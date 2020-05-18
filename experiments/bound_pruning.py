from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import zero_one_loss, accuracy_score
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from copy import copy

from partitioning_machines import DecisionTreeClassifier, breiman_alpha_pruning_objective, gini_impurity_criterion, decision_tree_to_tikz, shawe_taylor_bound_pruning_objective_factory


def prune_with_bound(
        decision_tree,
        bound=None):
    
    best_bound = bound(decision_tree.tree)
    bounds_value = decision_tree.compute_pruning_coefficients(bound)
    
    while bounds_value[0] <= best_bound:
        best_bound = bounds_value[0]
        decision_tree.prune_tree(best_bound)
        bounds_value = decision_tree.compute_pruning_coefficients(bound)
    
    return best_bound
    

if __name__ == '__main__':
    decision_tree = DecisionTreeClassifier(gini_impurity_criterion)
    X, y = load_iris(return_X_y=True)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.4, random_state=42)
    n_examples, n_features = X.shape
    table = {}
    bound = shawe_taylor_bound_pruning_objective_factory(n_features, table)
    
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
    best_bound = prune_with_bound(decision_tree, bound)
    print(f'Best bound: {best_bound:.3f}')

    pruned_tree = decision_tree_to_tikz(decision_tree, decision_tree.label_encoder.labels)
    acc_tr = accuracy_score(y_true=y_tr, y_pred=decision_tree.predict(X_tr))
    acc_ts = accuracy_score(y_true=y_ts, y_pred=decision_tree.predict(X_ts))
    print(f'Accuracy score of pruned tree on train dataset: {acc_tr:.3f}')
    print(f'Accuracy score of pruned tree on test dataset: {acc_ts:.3f}')
    
    import python2latex as p2l

    doc = p2l.Document('bound_tree_pruning', path='./experiments/', doc_type='standalone', border='1cm')
    doc.add_package('tikz')
    del doc.packages['geometry']
    doc.add_to_preamble('\\usetikzlibrary{shapes}')
    doc += original_tree
    doc += r'\hspace{1cm}'
    doc += pruned_tree
    doc.build()