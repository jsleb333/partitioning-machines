from partitioning_machines.partitioning_function_upper_bound import PartitioningFunctionUpperBound


def vcdim_upper_bound(tree, n_features):
    """
    Computes an upper bound on the VC dimension of a tree knowing the number of available features. Implements Algorithm 2 of Appendix E of Leboeuf et al. (2020).

    Args:
        tree (Tree object): Tree structure for which to compute the bound.
        n_features (int): Number of real-valued features.. Corresponds to the variable '\ell' in the paper.
    """
    if tree.is_leaf():
        return 1

    m = tree.n_leaves + 1
    pfub = PartitioningFunctionUpperBound(tree, n_features)
    while pfub(m, 2) == 2**(m-1)-1:
        m += 1

    return m - 1


def vcdim_lower_bound(tree, n_features):
    """
    Computes a lower bound on the VC dimension of a tree knowing the number of available features. Implements Algorithm 3 of Appendix E of Leboeuf et al. (2020).

    Args:
        tree (Tree object): Tree structure for which to compute the bound.
        n_features (int): Number of real-valued features.. Corresponds to the variable '\ell' in the paper.
    """
    if tree.is_leaf():
        return 1
    if tree.is_stump():
        return vcdim_upper_bound(tree, n_features) # Upper bound is exact for stumps
    else:
        return vcdim_lower_bound(tree.left_subtree, n_features) + vcdim_lower_bound(tree.right_subtree, n_features)
