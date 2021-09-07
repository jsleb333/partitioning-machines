from partitioning_machines import PartitioningFunctionUpperBound


def vcdim_upper_bound(tree, n_rl_feat, ordinal_feat_dist=None, nominal_feat_dist=None):
    """
    Computes an upper bound on the VC dimension of a tree knowing the number of available features. Implements Algorithm 2 of Appendix E of Leboeuf et al. (2020).

    Args:
        tree (Tree object):
            Tree structure for which to compute the bound.
        n_rl_feat (int):
            Number of real-valued features. Corresponds to the variable '\ell' in the paper.
        ordinal_feat_dist (Union[Sequence[int], None]):
            Feature distribution of the ordinal features. If None, it is assumed no ordinal features are used. The feature distribution should have the form [o_1, o_2, ...], where o_C is the number of features with C categories (o_1 should be 0 and will ignored).
        nominal_feat_dist (Union[Sequence[int], None]):
            Feature distribution of the nominal features. If None, it is assumed no nominal features are used. The feature distribution should have the form [n_1, n_2, ...], where n_C is the number of features with C categories (n_1 should be 0 and will ignored).
    """
    if tree.is_leaf():
        return 1

    m = tree.n_leaves + 1
    pfub = PartitioningFunctionUpperBound(tree, n_rl_feat, ordinal_feat_dist=ordinal_feat_dist, nominal_feat_dist=nominal_feat_dist)
    while pfub(m, 2) == 2**(m-1)-1:
        m += 1

    return m - 1


def vcdim_lower_bound(tree, n_rl_feat):
    """
    Computes a lower bound on the VC dimension of a tree knowing the number of available features. Implements Algorithm 3 of Appendix E of Leboeuf et al. (2020).

    Args:
        tree (Tree object):
            Tree structure for which to compute the bound.
        n_rl_feat (int):
            Number of real-valued features. Corresponds to the variable '\ell' in the paper.
    """
    if tree.is_leaf():
        return 1
    if tree.is_stump():
        return vcdim_upper_bound(tree, n_rl_feat) # Upper bound is exact for stumps
    else:
        return vcdim_lower_bound(tree.left_subtree, n_rl_feat) + vcdim_lower_bound(tree.right_subtree, n_rl_feat)
