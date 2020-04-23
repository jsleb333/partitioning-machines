"""Implementation of a binary tree"""
import numpy as np
from scipy.special import binom
from sympy.functions.combinatorial.numbers import stirling


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
        self._n_leaves = None
        self._hash_value = None

    @property
    def n_leaves(self):
        if self._n_leaves is None:
            self.update_tree()
        return self._n_leaves

    def update_tree(self):
        if self.is_leaf():
            self._n_leaves = 1
            self._hash_value = 0
        else:
            self._n_leaves = self.left_subtree.update_tree() + self.right_subtree.update_tree()
            self._hash_value = self.n_leaves + self.left_subtree.hash_value + self.right_subtree.hash_value
        return self._n_leaves

    @property
    def hash_value(self):
        if self._hash_value is None:
            self.update_tree()
        return self._hash_value

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
        return self.hash_value
