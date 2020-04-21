"""Implementation of a binary tree"""
import numpy as np
from scipy.special import binom
from sympy.functions.combinatorial.numbers import stirling
from sympy.functions.combinatorial.factorials import ff

class Tree:
    def __init__(self, left_subtree=None, right_subtree=None, n_real_valued_features=1):
        """
        Args:
            left_subtree ():
            right_subtree ():
        """
        self.left_subtree = left_subtree
        self.right_subtree = right_subtree
        self.n_rv_feat = n_real_valued_features
        self.partitioning_value = {0:0, 1:0} # pi^2_T(0) = pi^2_T(1) = 0

    @property
    def n_leaves(self):
        if self.is_leaf():
            return 1
        else:
            return self.left_subtree.n_leaves + self.right_subtree.n_leaves

    def is_leaf(self):
        return self.left_subtree is None and self.right_subtree is None

    def is_stump(self):
        if self.is_leaf():
            return False
        return self.left_subtree.is_leaf() and self.right_subtree.is_leaf()

    def __eq__(self, other):
        if other is None:
            return False
        if self.is_leaf() and other.is_leaf():
            return True
        if self.is_stump() and other.is_stump():
            return True
        if (self.left_subtree == other.left_subtree and self.right_subtree == other.right_subtree) \
            or (self.left_subtree == other.right_subtree and self.right_subtree == other.left_subtree):
            return True
        else:
            return False

    def partitioning_function_upper_bound(self, c, m):
        if c >

        if self.is_leaf():
            return 1

        if m not in self.partitioning_value:
            if self.is_stump():
                value = 1/2 * sum(min(2*self.n_rv_feat, binom(m,k)) for k in range(1,m))
            else:
                Qc = lambda k:
                delta_lr = (self.left_subtree == self.right_subtree)

            self.partitioning_value[m] = value

        return self.partitioning_value[m]
