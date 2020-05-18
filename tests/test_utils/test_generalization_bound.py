from sklearn.datasets import load_iris

from partitioning_machines.utils.generalization_bounds import *
from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion


def test_shawe_taylor_bound_pruning_objective():
    X, y = load_iris(return_X_y=True)
    dtc = DecisionTreeClassifier(gini_impurity_criterion)
    dtc.fit(X, y)
    
    pruning_objective = shawe_taylor_bound_pruning_objective_factory(10)
    pruning_objective(dtc.tree)
    pruning_objective(dtc.tree.left_subtree)
    