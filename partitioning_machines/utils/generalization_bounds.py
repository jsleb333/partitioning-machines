import numpy as np
from copy import copy
from scipy.special import zeta
from partitioning_machines import growth_function_upper_bound


def shawe_taylor_bound(n_examples, n_errors, growth_function, hypothesis_class_index, delta=.05):
    """
    Theorem 2.3 of Shawe-Taylor et al. (1997), Structural Risk Minimization over Data-Dependent Hierarchies, with the modification that Sauer's lemma is not used.
    """
    s = 2
    p_d = 1/zeta(s) * 1/(hypothesis_class_index + 1)**2
    r = 1/2
    q_k = (1-r) * r**n_errors
    # q_k = 1/zeta(s) * 1/(n_errors + 1)**2
    epsilon = 2*n_errors + 4*np.log(float(growth_function(2*n_examples))) + 4*np.log(4/(p_d*q_k*delta))
    return epsilon / n_examples

def shawe_taylor_bound_pruning_objective_factory(n_features, table={}, loose_pfub=True):
    def shawe_taylor_bound_pruning_objective(subtree):
        copy_of_tree = copy(subtree.root)
        copy_of_subtree = copy_of_tree.follow_path(subtree.path_from_root())
        copy_of_subtree.remove_subtree()
        
        n_classes = copy_of_tree.n_examples_by_label.shape[0]
        growth_function = growth_function_upper_bound(copy_of_tree, n_features, n_classes, table, loose_pfub)
        n_examples = copy_of_tree.n_examples
        n_errors = copy_of_tree.n_errors
        hypothesis_class_index = copy_of_tree.hash_value
        
        return shawe_taylor_bound(n_examples, n_errors, growth_function, hypothesis_class_index)
    return shawe_taylor_bound_pruning_objective


def vapnik_bound(n_examples, n_errors, growth_function, hypothesis_class_index, delta=.05):
    """
    Equation (4.41) of Vapnik's book (1998) extended to SRM.
    """
    s = 2
    p_d = 1/zeta(s) * 1/(hypothesis_class_index + 1)**2
    r = 1/2000
    q_k = (1-r) * r**n_errors
    # q_k = 1/zeta(s) * 1/(n_errors + 1)**2
    epsilon = 4 / n_examples * (np.log(float(growth_function(2*n_examples))) + np.log(4/(p_d*q_k*delta)))
    
    empirical_risk = n_errors / n_examples
    
    return empirical_risk + epsilon/2 * (1 + np.sqrt(1 + 4*empirical_risk/epsilon))

def vapnik_bound_pruning_objective_factory(n_features, table={}, loose_pfub=True):
    def vapnik_bound_pruning_objective(subtree):
        copy_of_tree = copy(subtree.root)
        copy_of_subtree = copy_of_tree.follow_path(subtree.path_from_root())
        copy_of_subtree.remove_subtree()
        
        n_classes = copy_of_tree.n_examples_by_label.shape[0]
        growth_function = growth_function_upper_bound(copy_of_tree, n_features, n_classes, table, loose_pfub)
        n_examples = copy_of_tree.n_examples
        n_errors = copy_of_tree.n_errors
        hypothesis_class_index = copy_of_tree.hash_value
        
        return vapnik_bound(n_examples, n_errors, growth_function, hypothesis_class_index)
    return vapnik_bound_pruning_objective