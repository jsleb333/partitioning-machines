from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np

from partitioning_machines import Tree, tree_from_sklearn_decision_tree


def test_tree_from_sklearn_decision_tree_with_mock():
    class MockTree:
        pass
    tree_ = MockTree()
    tree_.children_left = np.array([1,3,-1,-1,-1])
    tree_.children_right = np.array([2,4,-1,-1,-1])
    mock_tree = MockTree()
    mock_tree.tree_ = tree_
    
    tree_from_sklearn = tree_from_sklearn_decision_tree(mock_tree)
    
    leaf = Tree()
    stump = Tree(leaf, leaf)
    assert tree_from_sklearn == Tree(stump, leaf)

def test_tree_from_sklearn_decision_tree_with_actual_tree():
    X, y = load_iris(return_X_y=True)
    sklearn_tree = DecisionTreeClassifier()
    sklearn_tree = sklearn_tree.fit(X, y)
    
    tree_from_sklearn = tree_from_sklearn_decision_tree(sklearn_tree)

    