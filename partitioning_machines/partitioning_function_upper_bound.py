from scipy.special import binom, factorial
from sympy.functions.combinatorial.numbers import stirling


class PartitioningFunctionUpperBound:
    """
    This class computes the partioning function upper bound of Theorem 14 of the paper of Leboeuf et al. (2020).

    It implements an optimized version of the algorithm 1 of Appendix E by avoiding to compute the same value for the same subtree structures inside the tree multiple times by storing already computed values.
    """
    def __init__(self, tree, n_features):
        self.tree = tree
        self.n_features = n_features

        self.subtrees = []
        self._compute_list_of_distinct_subtrees(tree)

        self.pfub_table = {subtree:{} for subtree in self.subtrees}

    def _compute_list_of_distinct_subtrees(self, tree):
        if not tree.is_leaf():
            self._compute_list_of_distinct_subtrees(tree.left_subtree)
            if tree.left_subtree not in self.subtrees:
                self.subtrees.append(tree.left_subtree)

            self._compute_list_of_distinct_subtrees(tree.right_subtree)
            if tree.right_subtree not in self.subtrees:
                self.subtrees.append(tree.right_subtree)

        if tree not in self.subtrees:
            self.subtrees.append(tree)

    def _compute_upper_bound(self, tree, n_parts, n_examples, n_features):
        # Modified version of Algorithm 1 of Appendix E of Leboeuf et al. (2020)
        c, m, l = n_parts, n_examples, n_features

        if c > m or c > tree.n_leaves:
            return 0
        elif c == m or c == 1 or m == 1:
            return 1
        elif m <= tree.n_leaves:
            return stirling(m, c)
        # Modification 1: Check first in the table if value is already computed.
        elif (c, m, l) not in self.pfub_table[tree]:
            N = 0
            min_k = tree.left_subtree.n_leaves
            max_k = m - tree.right_subtree.n_leaves
            for k in range(min_k, max_k+1):
                N += min(2*l, binom(m, k)) * sum(
                    sum(
                        binom(a, c - b) * binom(b, c - a) * factorial(a + b - c) *
                        self._compute_upper_bound(tree.left_subtree, a, k, l) *
                        self._compute_upper_bound(tree.right_subtree, b, m-k, l)
                        for b in range(max(1,c-a), c+1)
                    )
                    for a in range(1, c+1)
                )

            if tree.left_subtree == tree.right_subtree:
                N /= 2

            # Modification 2: Add value to look up table.
            self.pfub_table[tree][n_parts, n_examples, n_features] = min(N, stirling(n_examples, n_parts))

        return self.pfub_table[tree][n_parts, n_examples, n_features]

    def __call__(self, n_examples, n_parts=2):
        return self._compute_upper_bound(self.tree, n_parts, n_examples, self.n_features)


def partitioning_function_upper_bound(tree, n_parts, n_examples, n_features):
    """
    Args:
        tree (Tree object):
        n_parts (int): Number of parts in the partitions. Corresponds to 'c' in the paper.
        n_examples (int): Number of examples. Corresponds to 'm' in the paper.
        n_features (int): Number of real-valued features. Corresponds to '\ell' in the paper.
    """
    pfub = PartitioningFunctionUpperBound(tree, n_features)
    return pfub(n_examples, n_parts)
