import numpy as np
from copy import copy


def shawe_taylor_bound(n_examples, n_errors, growth_function, hypothesis_class_index, delta=.05):
    """
    Theorem 2.3 of Shawe-Taylor et al. (1997), Structural Risk Minimization over Data-Dependent Hierarchies, with the modification that Sauer's lemma is not used.
    """
    p_d = 6/(np.pi**2*(hypothesis_class_index + 1))
    q_k = 6/(np.pi**2*(n_errors + 1))
    epsilon = 2 * n_errors + 4 * np.log(growth_function(2 * n_examples) + np.log(4/(p_d*q_k*delta)))
    return epsilon / n_examples

def shawe_taylor_bound_pruning_objective(subtree):
    copy_of_subtree = copy(subtree)
    copy_of_subtree.left_subtree = None
    copy_of_subtree.right_subtree = None
    copy_of_subtree.tree_root.update_tree()