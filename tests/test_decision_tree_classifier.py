import numpy as np

from partitioning_machines.decision_tree_classifier import *
from partitioning_machines.decision_tree_classifier import _DecisionTree

n_examples = 5
n_features = 4
n_classes = 3
X = np.array([[1,2,3,4],
              [3,4,6,3],
              [6,7,3,2],
              [5,5,2,6],
              [9,1,9,5]
              ])
y = np.array([0,1,0,2,2])
encoded_y = np.array([[1,0,0],
                      [0,1,0],
                      [1,0,0],
                      [0,0,1],
                      [0,0,1]
                      ])
n_examples_by_label = np.array([2,1,2])
frac_examples_by_label = n_examples_by_label/5


def test_gini_criterion_standard():
    expected_impurity_score = 4/5*(1-2/5) + (1-1/5)/5
    assert gini_impurity_criterion(frac_examples_by_label) == expected_impurity_score
def test_gini_criterion_vectorized_features():
    frac_examples_by_label_vec = np.array([frac_examples_by_label]*n_features)
    assert gini_impurity_criterion(frac_examples_by_label_vec).shape == (n_features,)

def test_entropy_criterion():
    assert entropy_impurity_criterion(frac_examples_by_label) == np.log(5) - 4/5 * np.log(2)    
def test_entropy_criterion_vectorized_features():
    frac_examples_by_label_vec = np.array([frac_examples_by_label]*n_features)
    assert entropy_impurity_criterion(frac_examples_by_label_vec).shape == (n_features,)

def test_margin_criterion():
    assert margin_impurity_criterion(frac_examples_by_label) == 3/5    
def test_margin_criterion_vectorized_features():
    frac_examples_by_label_vec = np.array([frac_examples_by_label]*n_features)
    assert margin_impurity_criterion(frac_examples_by_label_vec).shape == (n_features,)
    

class TestSplitter:
    def test_find_best_split_at_init(self):
        tree = _DecisionTree(gini_impurity_criterion(frac_examples_by_label), n_examples_by_label)
        
        X_idx_sorted = np.argsort(X, 0)
        assert (X_idx_sorted == np.array([[0,4,3,2],
                                          [1,0,0,1],
                                          [3,1,2,0],
                                          [2,3,1,4],
                                          [4,2,4,3]
                                          ])).all()
        assert (encoded_y[X_idx_sorted[0]] == np.array([[1,0,0],
                                                        [0,0,1],
                                                        [0,0,1],
                                                        [1,0,0]
                                                        ])).all()
        assert (n_examples_by_label - encoded_y[X_idx_sorted[0]] == np.array([[1,1,2],
                                                                              [2,1,1],
                                                                              [2,1,1],
                                                                              [1,1,2]
                                                                              ])).all()
        
        split = Splitter(tree, X, encoded_y, X_idx_sorted, gini_impurity_criterion, 'min')
        assert split.rule_feature == 3
        assert split.rule_threshold == 4.5
        assert np.isclose(4/9 * 3/5, split.impurity_score)
        assert (split.n_examples_by_label_left == np.array([2,1,0])).all()
        assert (split.n_examples_by_label_right == np.array([0,0,2])).all()
    
    def test_argext_min(self):
        tree = _DecisionTree(gini_impurity_criterion(frac_examples_by_label), n_examples_by_label)
        split = Splitter(tree, X, encoded_y, np.argsort(X, 0), gini_impurity_criterion, 'min')
        idx, opt = split.argext(np.array([4,2,3,4]))
        assert idx == 1
        assert opt == 2
        
    def test_argext_max(self):
        tree = _DecisionTree(gini_impurity_criterion(frac_examples_by_label), n_examples_by_label)
        split = Splitter(tree, X, encoded_y, np.argsort(X, 0), gini_impurity_criterion, 'max')
        idx, opt = split.argext(np.array([4,2,3,4]))
        assert idx == 0
        assert opt == 4
        
    def test_split_impurity_criterion(self):
        X_idx_sorted = np.argsort(X, 0)
        tree = _DecisionTree(gini_impurity_criterion(frac_examples_by_label), n_examples_by_label)
        split = Splitter(tree, X, encoded_y, X_idx_sorted, gini_impurity_criterion, 'min')

        n_examples_by_label_left = encoded_y[X_idx_sorted[0]] # Shape: (n_features, n_classes)
        n_examples_by_label_right = n_examples_by_label - n_examples_by_label_left
        
        split_impurity = split._split_impurity_criterion(n_examples_by_label_left, n_examples_by_label_right, 1, 4)
        expected_split_impurity = 4/5*np.array(gini_impurity_criterion(n_examples_by_label_right/4))
        assert np.isclose(split_impurity, expected_split_impurity).all()
    
    def test_split_makes_gain(self):
        tree = _DecisionTree(gini_impurity_criterion(frac_examples_by_label), n_examples_by_label)
        split = Splitter(tree, X, encoded_y, np.argsort(X, 0), gini_impurity_criterion, 'min')
        assert split.split_makes_gain()
        split.impurity_score = 1
        assert not split.split_makes_gain()
    
    def test_compute_split_X_idx_sorted(self):
        tree = _DecisionTree(gini_impurity_criterion(frac_examples_by_label), n_examples_by_label)
        split = Splitter(tree, X, encoded_y, np.argsort(X, 0), gini_impurity_criterion, 'min')
        X_idx_sorted_left, X_idx_sorted_right = split._compute_split_X_idx_sorted()
        assert (X_idx_sorted_left == np.array([[0,0,0,2],
                                               [1,1,2,1],
                                               [2,2,1,0]])).all()
        assert (X_idx_sorted_right == np.array([[3,4,3,4],
                                                [4,3,4,3]])).all()
    
    def test_apply_split(self):
        tree = _DecisionTree(gini_impurity_criterion(frac_examples_by_label), n_examples_by_label)
        split = Splitter(tree, X, encoded_y, np.argsort(X, 0), gini_impurity_criterion, 'min')
        split.apply_split()
        assert tree.left_subtree is not None
        assert tree.right_subtree is not None
        assert tree.rule_feature == 3
        assert tree.rule_threshold == 4.5
        
        assert np.isclose(tree.left_subtree.impurity_score, 4/9)
        assert (tree.left_subtree.label == np.array([1,0,0])).all()
        assert tree.right_subtree.impurity_score == 0
        assert (tree.right_subtree.label == np.array([0,0,1])).all()
        
    def test_leaves_splitter(self):
        tree = _DecisionTree(gini_impurity_criterion(frac_examples_by_label), n_examples_by_label)
        split = Splitter(tree, X, encoded_y, np.argsort(X, 0), gini_impurity_criterion, 'min')
        split.apply_split()
        splits = split.leaves_splitters()
        assert tree.left_subtree is splits[0].leaf
    
    def test_n_examples_left_right(self):
        tree = _DecisionTree(gini_impurity_criterion(frac_examples_by_label), n_examples_by_label)
        split = Splitter(tree, X, encoded_y, np.argsort(X, 0), gini_impurity_criterion, 'min')
        assert split.n_examples_left == 3
        assert split.n_examples_right == 2

    
class TestDecisionTreeClassifier:
    def test_init_tree(self):
        dtc = DecisionTreeClassifier(gini_impurity_criterion)
        dtc._init_tree(encoded_y, n_examples)
        assert dtc.tree.impurity_score == gini_impurity_criterion(frac_examples_by_label)
        assert all(dtc.tree.n_examples_by_label == n_examples_by_label)
    
    def test_fit_max_2_leaves(self):
        dtc = DecisionTreeClassifier(gini_impurity_criterion, max_n_leaves=2)
        dtc.fit(X, y)
        assert dtc.tree.height == 1
        assert dtc.tree.n_leaves == 2
        assert np.isclose(dtc.tree.left_subtree.impurity_score, 4/9)
        assert (dtc.tree.left_subtree.label == np.array([1,0,0])).all()
        assert dtc.tree.right_subtree.impurity_score == 0
        assert (dtc.tree.right_subtree.label == np.array([0,0,1])).all()
    
    def test_fit_no_max_leaves(self):
        dtc = DecisionTreeClassifier(gini_impurity_criterion)
        dtc.fit(X, y)
        assert dtc.tree.n_leaves == 3
        assert all(leaf.is_pure() for leaf in dtc.tree if leaf.is_leaf())

    def test_fit_max_depth(self):
        dtc = DecisionTreeClassifier(gini_impurity_criterion, max_depth=1)
        dtc.fit(X, y)
        assert dtc.tree.height == 1
        assert dtc.tree.n_leaves == 2
        assert np.isclose(dtc.tree.left_subtree.impurity_score, 4/9)
        assert (dtc.tree.left_subtree.label == np.array([1,0,0])).all()
        assert dtc.tree.right_subtree.impurity_score == 0
        assert (dtc.tree.right_subtree.label == np.array([0,0,1])).all()
    
    def test_fit_min_2_examples_per_leaf(self):
        dtc = DecisionTreeClassifier(gini_impurity_criterion, min_examples_per_leaf=2)
        dtc.fit(X, y)
        assert dtc.tree.height == 1
        assert dtc.tree.n_leaves == 2
        assert np.isclose(dtc.tree.left_subtree.impurity_score, 4/9)
        assert (dtc.tree.left_subtree.label == np.array([1,0,0])).all()
        assert dtc.tree.right_subtree.impurity_score == 0
        assert (dtc.tree.right_subtree.label == np.array([0,0,1])).all()
    