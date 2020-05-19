from sklearn.datasets import load_iris

from partitioning_machines.utils.generalization_bounds import *
from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion


def test_shawe_taylor_bound_pruning_objective():
    X, y = load_iris(return_X_y=True)
    n_features = X.shape[1] # = 4
    dtc = DecisionTreeClassifier(gini_impurity_criterion)
    dtc.fit(X[40:90], y[40:90])
    
    table = {}
    pruning_objective = shawe_taylor_bound_pruning_objective_factory(n_features, table)
    pruning_objective(dtc.tree.left_subtree)
    assert table

def test_vapnik_bound_pruning_objective():
    X, y = load_iris(return_X_y=True)
    n_features = X.shape[1] # = 4
    dtc = DecisionTreeClassifier(gini_impurity_criterion)
    dtc.fit(X[40:90], y[40:90])
    
    table = {}
    pruning_objective = vapnik_bound_pruning_objective_factory(n_features, table)
    pruning_objective(dtc.tree.left_subtree)
    assert table