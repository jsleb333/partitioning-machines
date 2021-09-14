from math import floor
from numpy import pi
from scipy.special import binom, factorial
from sympy.functions.combinatorial.numbers import stirling
from sympy.functions.combinatorial.factorials import ff
from copy import copy


class PartitioningFunctionUpperBound:
    """
    This class computes the partioning function upper bound of Theorem 9 of the paper 'Decision trees as partitioning machines to characterize their generalization properties' by Leboeuf, LeBlanc and Marchand (2020).

    It implements an optimized version of the algorithm 1 of Appendix D by avoiding to compute the same value for the same subtree structures inside the tree multiple times by storing already computed values.
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
        n = copy(nominal_feat_dist)
        while n[-1] == 0 and len(n) > 2:
            del n[-1]

        if n_examples > len(n) or len(n) == 2:
            return n

        cumul = 0
        for n_C in n[n_examples:]:
            cumul += n_C

        n[n_examples-1] += cumul
        del n[n_examples:]

        return n

    def _update_nominal_feat_dist(self, nominal_feat_dist, C):
        new_dist = copy(nominal_feat_dist)
        new_dist[C] -= 1
        new_dist[C-1] += 1
        return new_dist

    def _check_trivial_cases(self, n_examples, n_parts, n_leaves):
        if n_parts > n_examples or n_parts > n_leaves:
            return 0
        elif n_parts == n_examples or n_parts == 1 or n_examples == 1:
            return 1
        elif n_examples <= n_leaves:
            return stirling(n_examples, n_parts)
        else:
            return None

    def _compute_upper_bound_tight(self,
                                   tree,
                                   n_parts,
                                   n_examples,
                                   nominal_feat_dist):
        """
        Optimized implementation of Algorithm 1 of Appendix D of the paper.
        """
        c, m, l, o = n_parts, n_examples, self.n_rl_feat, self.ordinal_feat_dist
        n = self._truncate_nominal_feat_dist(nominal_feat_dist, m)

        trivial_case = self._check_trivial_cases(m, c, tree.n_leaves)
        if trivial_case is not None:
            return trivial_case

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
                    coef_ord = min(2*l + omega + o[1], binom(m, k))
                    coef_nom = lambda nC, C: nC * min(max(2, C), floor(m/min(k, m-k)))
                else:
                    coef_ord = min(2*l + omega, binom(m, k))
                    coef_nom = lambda nC, C: nC * min(C, floor(m/min(k, m-k)))

                # if c == 2: # Algo simplified for c = 2 (is faster)
                #     N +=  min(2*l + omega_k, binom(m, k)) * (1
                #             + 2 * self._compute_upper_bound_tight(tree.left_subtree, 2, k, n)
                #             + 2 * self._compute_upper_bound_tight(tree.right_subtree, 2, m-k, n)
                #             + 2 * self._compute_upper_bound_tight(tree.left_subtree, 2, k, n) * self._compute_upper_bound_tight(tree.right_subtree, 2, m-k, n)
                #             )

                # Modification 2: To avoid making useless calls, we expand some cases of a and b that can be simplified
                tmp_ord = 0
                tmp_nom = 0
                for a in range(1, c+1):
                    for b in range(max(1, c-a), c+1):
                        if a == 1 and b == 1:
                            tmp_ord += 1
                            tmp_nom += sum(
                                coef_nom(n[C], C) for C in range(1, len(n)) if n[C] > 0
                            )
                        elif a == 1 and b == c-1:
                            tmp_ord += self._compute_upper_bound_tight(tree.right_subtree, b, m-k, n)
                            tmp_nom += sum(
                                coef_nom(n[C], C) * self._compute_upper_bound_tight(tree.right_subtree, b, m-k, self._update_nominal_feat_dist(n, C))
                                for C in range(1, len(n)) if n[C] > 0
                            )
                        elif a == 1 and b == c:
                            tmp_ord += c*self._compute_upper_bound_tight(tree.right_subtree, b, m-k, n)
                            tmp_nom += c*sum(
                                coef_nom(n[C], C) * self._compute_upper_bound_tight(tree.right_subtree, b, m-k, self._update_nominal_feat_dist(n, C))
                                for C in range(1, len(n)) if n[C] > 0
                            )
                        elif b == 1 and a == c-1:
                            tmp_ord += self._compute_upper_bound_tight(tree.left_subtree, a, k, n)
                            tmp_nom += sum(
                                coef_nom(n[C], C) * self._compute_upper_bound_tight(tree.left_subtree, a, k, self._update_nominal_feat_dist(n, C))
                                for C in range(1, len(n)) if n[C] > 0
                            )
                        elif b == 1 and a == c:
                            tmp_ord += c*self._compute_upper_bound_tight(tree.left_subtree, a, k, n)
                            tmp_nom += c*sum(
                                coef_nom(n[C], C) * self._compute_upper_bound_tight(tree.left_subtree, a, k, self._update_nominal_feat_dist(n, C))
                                for C in range(1, len(n)) if n[C] > 0
                            )
                        else:
                            pi_left = self._check_trivial_cases(k, a, tree.left_subtree.n_leaves)
                            pi_right = self._check_trivial_cases(m-k, b, tree.right_subtree.n_leaves)

                            if pi_left == 0 or pi_right == 0:
                                continue

                            Kabc = binom(a, c - b) * binom(b, c - a) * factorial(a + b - c)

                            # Ordinal first
                            pi_left_ord = pi_left or self._compute_upper_bound_tight(tree.left_subtree, a, k, n)
                            pi_right_ord = pi_right or self._compute_upper_bound_tight(tree.right_subtree, b, m-k, n)

                            tmp_ord += Kabc * pi_left_ord * pi_right_ord

                            # Nominal second
                            if pi_left is None and pi_right is not None:
                                pi_left_nom = sum(
                                    coef_nom(n[C], C) * self._compute_upper_bound_tight(tree.left_subtree, a, k, self._update_nominal_feat_dist(n, C))
                                    for C in range(1, len(n)) if n[C] > 0
                                )
                                tmp_nom += Kabc * pi_left_nom * pi_right
                            elif pi_left is not None and pi_right is None:
                                pi_right_nom = sum(
                                    coef_nom(n[C], C) * self._compute_upper_bound_tight(tree.right_subtree, b, m-k, self._update_nominal_feat_dist(n, C))
                                    for C in range(1, len(n)) if n[C] > 0
                                )
                                tmp_nom += Kabc * pi_left * pi_right_nom
                            elif pi_left is None and pi_right is None:
                                tmp_nom += Kabc * sum(
                                    coef_nom(n[C], C)
                                    * self._compute_upper_bound_tight(tree.left_subtree, a, k, self._update_nominal_feat_dist(n, C))
                                    * self._compute_upper_bound_tight(tree.right_subtree, b, m-k, self._update_nominal_feat_dist(n, C))
                                    for C in range(1, len(n)) if n[C] > 0
                                )
                            else:
                                tmp_nom += Kabc * pi_left * pi_right * sum(
                                        coef_nom(n[C], C) for C in range(1, len(n)) if n[C] > 0
                                    )

                N += coef_ord * tmp_ord + tmp_nom

            if tree.left_subtree == tree.right_subtree:
                N /= 2

            # Modification 3: Add value to lookup table.
            self.pfub_table[tree][c, m, l, tuple(n[1:])] = min(N, stirling(n_examples, n_parts))

        return self.pfub_table[tree][c, m, l, tuple(n[1:])]

    def _compute_upper_bound_loose(self,
                                   tree,
                                   n_parts,
                                   n_examples,
                                   nominal_feat_dist):
        """
        Looser but faster implementation of Algorithm 1 of Appendix D of the paper. The corresponding equation can be found at the end of section 6.2.
        """
        c, m, l = n_parts, n_examples, self.n_rl_feat
        n = self._truncate_nominal_feat_dist(nominal_feat_dist, m)

        trivial_case = self._check_trivial_cases(m, c, tree.n_leaves)
        if trivial_case is not None:
            return trivial_case

        # Modification 1: Check first in the table if value is already computed.
        if tree not in self.pfub_table:
            self.pfub_table[tree] = {}
        if (c, m, l, tuple(n[1:])) not in self.pfub_table[tree]:
            k_left = m - tree.right_subtree.n_leaves
            k_right = m - tree.left_subtree.n_leaves


            coef_nom = min(
                max(floor(m/min(k_left, m-k_left)), floor(m/min(k_right, m-k_right)))*self.n_nominal_feat,
                sum(C*n[C] for C in range(1, len(n)))
            )
            coef = (m - tree.n_leaves) * (2*l + 2*self.n_ordinal_feat + coef_nom)

            # Modification 2: To avoid making useless calls, we expand some cases of a and b that can be simplified
            N = 0
            for a in range(1, c+1):
                for b in range(max(1, c-a), c+1):
                    if a == 1 and b == 1:
                        N += 1
                    elif a == 1 and b == c-1:
                        N += self._compute_upper_bound_loose(tree.right_subtree, b, k_right, n)
                    elif a == 1 and b == c:
                        N += c*self._compute_upper_bound_loose(tree.right_subtree, b, k_right, n)
                    elif b == 1 and a == c-1:
                        N += self._compute_upper_bound_loose(tree.left_subtree, a, k_left, n)
                    elif b == 1 and a == c:
                        N += c*self._compute_upper_bound_loose(tree.left_subtree, a, k_left, n)
                    else:
                        pi_left = self._check_trivial_cases(k_left, a, tree.left_subtree.n_leaves)
                        pi_right = self._check_trivial_cases(k_right, b, tree.right_subtree.n_leaves)

                        if pi_left == 0 or pi_right == 0:
                            continue

                        if pi_left is None:
                            pi_left = self._compute_upper_bound_loose(tree.left_subtree, a, k_left, n)
                        if pi_right is None:
                            pi_right = self._compute_upper_bound_loose(tree.right_subtree, b, k_right, n)

                        Kabc = binom(a, c - b) * binom(b, c - a) * factorial(a + b - c)
                        N += Kabc * pi_left * pi_right

            N *= coef

            if tree.left_subtree == tree.right_subtree:
                N /= 2

            # Modification 3: Add value to lookup table.
            self.pfub_table[tree][c, m, l, tuple(n[1:])] = min(N, stirling(n_examples, n_parts))

        return self.pfub_table[tree][c, m, l, tuple(n[1:])]

    def __call__(self, n_examples, n_parts=2):
        """
        Args:
            n_examples (int): Number of examples. Corresponds to the variable 'm' in the paper.
            n_parts (int): Number of parts. Corresponds to the variable 'c' in the paper.
        """
        if self.loose:
            return self._compute_upper_bound_loose(self.tree, n_parts, n_examples, nominal_feat_dist=self.nominal_feat_dist)
        else:
            return self._compute_upper_bound_tight(self.tree, n_parts, n_examples, nominal_feat_dist=self.nominal_feat_dist)


def partitioning_function_upper_bound(tree,
                                      n_parts,
                                      n_examples,
                                      n_rl_feat,
                                      ordinal_feat_dist=None,
                                      nominal_feat_dist=None,
                                      pre_computed_tables=None,
                                      loose=False):
    r"""
    Args:
        tree (Tree object):
            Tree structure for which to compute the bound.
        n_parts (int):
            Number of parts in the partitions. Corresponds to the variable 'c' in the paper.
        n_examples (int):
            Number of examples. Corresponds to the variable 'm' in the paper.
        n_rl_feat (int):
            Number of real-valued features. Corresponds to the variable '\ell' in the paper.
    """
    pfub = PartitioningFunctionUpperBound(tree,
                                          n_rl_feat,
                                          ordinal_feat_dist=ordinal_feat_dist,
                                          nominal_feat_dist=nominal_feat_dist,
                                          pre_computed_tables=pre_computed_tables,
                                          loose=loose)
    return pfub(n_examples, n_parts)


def growth_function_upper_bound(tree,
                                n_rl_feat,
                                ordinal_feat_dist=None,
                                nominal_feat_dist=None,
                                n_classes=2,
                                pre_computed_tables=None,
                                loose=False):
    pfub = PartitioningFunctionUpperBound(tree,
                                          n_rl_feat,
                                          ordinal_feat_dist=ordinal_feat_dist,
                                          nominal_feat_dist=nominal_feat_dist,
                                          pre_computed_tables=pre_computed_tables,
                                          loose=loose)
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
    tree = Tree(stump, stump)
    c = 2
    m = 50
    for _ in Timer(range(1)):
        pfub = PartitioningFunctionUpperBound(tree, 10, nominal_feat_dist=[0,0,0], loose=False)
        print(pfub(m, c))
