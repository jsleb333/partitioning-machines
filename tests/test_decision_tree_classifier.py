import numpy as np

from partitioning_machines.decision_tree_classifier import *

n_examples = 5
n_features = 4
n_classes = 3
X = np.array([[1,2,3,4],
              [3,4,6,3],
              [1,7,3,2],
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
frac_examples_by_label = np.array([2,1,2])/5


def test_gini_criterion_standard():
    assert gini_impurity_criterion(frac_examples_by_label) == 4/5*(1-2/5) + (1-1/5)/5
def test_gini_criterion_vectorized_features():
    frac_examples_by_label_vec = np.array([frac_examples_by_label]*n_features).T
    assert gini_impurity_criterion(frac_examples_by_label_vec).shape == (n_features,)

def test_entropy_criterion():
    assert entropy_impurity_criterion(frac_examples_by_label) == np.log(5) - 4/5 * np.log(2)    
def test_entropy_criterion_vectorized_features():
    frac_examples_by_label_vec = np.array([frac_examples_by_label]*n_features).T
    assert entropy_impurity_criterion(frac_examples_by_label_vec).shape == (n_features,)

def test_margin_criterion():
    assert margin_impurity_criterion(frac_examples_by_label) == 3/5    
def test_margin_criterion_vectorized_features():
    frac_examples_by_label_vec = np.array([frac_examples_by_label]*n_features).T
    assert margin_impurity_criterion(frac_examples_by_label_vec).shape == (n_features,)