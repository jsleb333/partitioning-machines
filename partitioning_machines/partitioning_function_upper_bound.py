from math import floor
from scipy.special import binom, factorial
from sympy.functions.combinatorial.numbers import stirling
from sympy.functions.combinatorial.factorials import ff
from copy import copy


class PartitioningFunctionUpperBound:
    """
    This class computes the partioning function upper bound of Theorem 14 of the paper of Leboeuf et al. (2020).

    It implements an optimized version of the algorithm 1 of Appendix E by avoiding to compute the same value for the same subtree structures inside the tree multiple times by storing already computed values.
    """
    def __init__(self, tree, n_rl_feat, *, ordinal_feat_dist=None, nominal_feat_dist=None, pre_computed_tables=None, loose=False):
        r"""
        Args:
            tree (Tree object):
                Tree structure for which to compute the bound.
            n_rl_feat (int):
                Number of real-valued features. Corresponds to the variable '\ell' in the paper.
            ordinal_feat_dist (Union[Sequence[int], None]):
                Feature distribution of the ordinal features. If None, it is assumed no ordinal features are used. The feature distribution should have the form [o_1, o_2, ...], where o_C is the number of features with C categories (o_1 should be 0 and will ignored).
            nominal_feat_dist (Union[Sequence[int], None]):
                Feature distribution of the nominal features. If None, it is assumed no nominal features are used. The feature distribution should have the form [n_1, n_2, ...], where n_C is the number of features with C categories (n_1 should be 0 and will ignored).
            pre_computed_tables (Union[dict, None]):
                If the upper bound has already been computed for another tree, the computed tables of the PartitioningFunctionUpperBound object can be transfered here to speed up the process for current tree. The transfered table will be updated with any new value computed. If None, a table will be created from scratch. One can get the computed table by accessing the 'pfub_table' attribute.
            loose (bool):
                If loose is True, a looser but *much more* computationally efficient version of the bound is computed. In that case, no table is needed.
        """
        self.tree = tree
        self.n_rl_feat = n_rl_feat

        self.nominal_feat_dist = nominal_feat_dist or [0, 0]
        self.n_nominal_feat = sum(self.nominal_feat_dist[1:])

        self.ordinal_feat_dist = ordinal_feat_dist or [0, 0]
        self.n_ordinal_feat = sum(self.ordinal_feat_dist[1:])

        self.pfub_table = {} if pre_computed_tables is None else pre_computed_tables
        self.loose = loose

    def _truncate_nominal_feat_dist(self, nominal_feat_dist, n_examples):
        # Remove trailing zeros
        while nominal_feat_dist[-1] == 0 and len(nominal_feat_dist) > 2:
            del nominal_feat_dist[-1]

        if n_examples > len(nominal_feat_dist) or len(nominal_feat_dist) == 2:
            return nominal_feat_dist

        cumul = 0
        for n_C in nominal_feat_dist[n_examples:]:
            cumul += n_C

        nominal_feat_dist[n_examples-1] += cumul
        del nominal_feat_dist[n_examples:]

        return nominal_feat_dist

    def _compute_upper_bound_tight(self,
                                   tree,
                                   n_parts,
                                   n_examples,
                                   nominal_feat_dist):
        """
        Optimized implementation of Algorithm 1 of Appendix E of Leboeuf et al. (2020).
        """
        c, m, l, o = n_parts, n_examples, self.n_rl_feat, self.ordinal_feat_dist
        n = self._truncate_nominal_feat_dist(nominal_feat_dist, m)

        if c > m or c > tree.n_leaves:
            return 0
        elif c == m or c == 1 or m == 1:
            return 1
        elif m <= tree.n_leaves:
            return stirling(m, c)
        # Modification 1: Check first in the table if value is already computed.
        if tree not in self.pfub_table:
            self.pfub_table[tree] = {}
        if (c, m, l, tuple(n[1:])) not in self.pfub_table[tree]:
            N = 0
            min_k = tree.left_subtree.n_leaves
            max_k = m - tree.right_subtree.n_leaves

            omega = 2*self.n_ordinal_feat - o[1]

            for k in range(min_k, max_k+1):
                if k == m/2:
                    omega_k = omega + o[1]
                else:
                    omega_k = omega

                # Modification 2: Since c = 2 is the most common use case, we give an optimized version, writing explicitely the sum over a and b.
                if False:#c == 2:
                    N +=  min(2*l + omega_k, binom(m, k)) * (1
                            + 2 * self._compute_upper_bound_tight(tree.left_subtree, 2, k, n)
                            + 2 * self._compute_upper_bound_tight(tree.right_subtree, 2, m-k, n)
                            + 2 * self._compute_upper_bound_tight(tree.left_subtree, 2, k, n) * self._compute_upper_bound_tight(tree.right_subtree, 2, m-k, n)
                            )

                    for C in range(1, len(n)-1):
                        if n[C] > 0:
                            new_dist = copy(n)
                            new_dist[C] -= 1
                            new_dist[C-1] += 1
                            N += n[C] * min(C, floor(m/min(k, m-k))) * (1
                                + 2 * self._compute_upper_bound_tight(tree.left_subtree, 2, k, new_dist)
                                + 2 * self._compute_upper_bound_tight(tree.right_subtree, 2, m-k, new_dist)
                                + 2 * self._compute_upper_bound_tight(tree.left_subtree, 2, k, new_dist) * self._compute_upper_bound_tight(tree.right_subtree, 2, m-k, new_dist)
                                )

                else:
                    # N += min(2*l + omega_k, binom(m, k)) * sum(
                    #     sum(
                    #         binom(a, c - b) * binom(b, c - a) * factorial(a + b - c) *
                    #         self._compute_upper_bound_tight(tree.left_subtree, a, k, n) *
                    #         self._compute_upper_bound_tight(tree.right_subtree, b, m-k, n)
                    #         for b in range(max(1,c-a), c+1)
                    #     )
                    #     for a in range(1, c+1)
                    # )
                    print(f'{k=}')
                    tmp = 0
                    for a in range(1, c+1):
                        for b in range(max(1, c-a), c+1):
                            print(f'{a=}, {b=}')
                            if a == 1 and b == 1:
                                tmp += 1
                            elif a == 1 and b == c-1:
                                tmp += self._compute_upper_bound_tight(tree.right_subtree, b, m-k, n)
                            elif a == 1 and b == c:
                                tmp += c*self._compute_upper_bound_tight(tree.right_subtree, b, m-k, n)
                            elif b == 1 and a == c-1:
                                tmp += self._compute_upper_bound_tight(tree.left_subtree, a, k, n)
                            elif b == 1 and a == c:
                                tmp += c*self._compute_upper_bound_tight(tree.left_subtree, a, k, n)
                            else:
                                print(f'{tmp=} before')
                                coef = binom(a, c - b) * binom(b, c - a) * factorial(a + b - c)
                                tmp += coef * self._compute_upper_bound_tight(tree.left_subtree, a, k, n) * self._compute_upper_bound_tight(tree.right_subtree, b, m-k, n)
                                print(f'{tmp=} after\n')

                    N += min(2*l + omega_k, binom(m, k)) * tmp

                    for C in range(1, len(n)-1):
                        if n[C] > 0:
                            new_dist = copy(n)
                            new_dist[C] -= 1
                            new_dist[C-1] += 1
                            N += n[C] * min(C, floor(m/min(k, m-k))) * sum(
                                sum(
                                    binom(a, c - b) * binom(b, c - a) * factorial(a + b - c) *
                                    self._compute_upper_bound_tight(tree.left_subtree, a, k, new_dist) *
                                    self._compute_upper_bound_tight(tree.right_subtree, b, m-k, new_dist)
                                    for b in range(max(1,c-a), c+1)
                                )
                                for a in range(1, c+1)
                            )

            if tree.left_subtree == tree.right_subtree:
                N /= 2

            # Modification 3: Add value to lookup table.
            self.pfub_table[tree][c, m, l, tuple(n[1:])] = min(N, stirling(n_examples, n_parts))

        return self.pfub_table[tree][c, m, l, tuple(n[1:])]

    def _compute_upper_bound_loose(self, tree, n_parts, n_examples):
        """
        Looser but faster implementation of Algorithm 1 of Appendix E of Leboeuf et al. (2020).
        """
        c, m, l = n_parts, n_examples, self.n_rl_feat

        if c > m or c > tree.n_leaves:
            return 0
        elif c == m or c == 1 or m == 1:
            return 1
        elif m <= tree.n_leaves:
            return stirling(m, c)
        if tree not in self.pfub_table:
            self.pfub_table[tree] = {}
        if (c, m, l) not in self.pfub_table[tree]:
            N = 0
            k_left = m - tree.right_subtree.n_leaves
            k_right = m - tree.left_subtree.n_leaves
            N = 0
            if c == 2:
                N +=  2*l * (1
                        + 2 * self._compute_upper_bound_loose(tree.left_subtree, 2, k_left)
                        + 2 * self._compute_upper_bound_loose(tree.right_subtree, 2, k_right)
                        + 2 * self._compute_upper_bound_loose(tree.left_subtree, 2, k_left) * self._compute_upper_bound_loose(tree.right_subtree, 2, k_right)
                        )
            else:
                N += 2*l * sum(
                    sum(
                        binom(a, c - b) * binom(b, c - a) * factorial(a + b - c) *
                        self._compute_upper_bound_loose(tree.left_subtree, a, k_left) *
                        self._compute_upper_bound_loose(tree.right_subtree, b, k_right)
                        for b in range(max(1,c-a), c+1)
                    )
                    for a in range(1, c+1)
                )
            N *= m - tree.n_leaves

            if tree.left_subtree == tree.right_subtree:
                N /= 2

            self.pfub_table[tree][c, m, l] = min(N, stirling(n_examples, n_parts))
        return self.pfub_table[tree][c, m, l]

    def __call__(self, n_examples, n_parts=2):
        """
        Args:
            n_examples (int): Number of examples. Corresponds to the variable 'm' in the paper.
            n_parts (int): Number of parts. Corresponds to the variable 'c' in the paper.
        """
        if self.loose:
            return self._compute_upper_bound_loose(self.tree, n_parts, n_examples)
        else:
            return self._compute_upper_bound_tight(self.tree, n_parts, n_examples, nominal_feat_dist=self.nominal_feat_dist)



def partitioning_function_upper_bound(tree, n_parts, n_examples, n_rl_feat):
    r"""
    Args:
        tree (Tree object): Tree structure for which to compute the bound.
        n_parts (int): Number of parts in the partitions. Corresponds to the variable 'c' in the paper.
        n_examples (int): Number of examples. Corresponds to the variable 'm' in the paper.
        n_rl_feat (int): Number of real-valued features. Corresponds to the variable '\ell' in the paper.
    """
    pfub = PartitioningFunctionUpperBound(tree, n_rl_feat)
    return pfub(n_examples, n_parts)


def growth_function_upper_bound(tree, n_rl_feat, n_classes=2, pre_computed_tables=None, loose=False):
    pfub = PartitioningFunctionUpperBound(tree, n_rl_feat, pre_computed_tables=pre_computed_tables, loose=loose)
    def upper_bound(n_examples):
        max_range = min(n_classes, tree.n_leaves, n_examples)
        return sum(ff(n_classes, n)*pfub(n_examples, n) for n in range(1, max_range+1))
    return upper_bound



if __name__ == '__main__':
    import os, sys
    sys.path.append(os.getcwd())
    from partitioning_machines import Tree
    from graal_utils import Timer

    leaf = Tree()
    stump = Tree(leaf, leaf)
    tree = Tree(stump, leaf)
    c = 2
    for m in Timer(range(1)):
        pfub = PartitioningFunctionUpperBound(tree, 50)
        print(pfub(5, c))
