"""Implementation of a binary tree"""
import numpy as np
from scipy.special import binom
from sympy.functions.combinatorial.numbers import stirling
from sympy.functions.combinatorial.factorials import ff

class Tree:
    def __init__(self, left_subtree=None, right_subtree=None):
        """
        Args:
            left_subtree ():
            right_subtree ():
        """
        if left_subtree is None and right_subtree is not None or left_subtree is not None and right_subtree is None:
            raise ValueError('Both subtrees must be either None or other valid trees.')

        self.left_subtree = left_subtree
        self.right_subtree = right_subtree

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

    def __hash__(self):
        # The hash is the sum of the depths d_i of each leaf i.
        return self._hash_func(0)

    def _hash_func(self, depth):
        if self.is_leaf():
            return depth
        else:
            return self.left_subtree._hash_func(depth+1) + self.right_subtree._hash_func(depth+1)
