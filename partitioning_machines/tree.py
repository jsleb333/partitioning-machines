"""Implementation of a binary tree"""
import numpy as np
from scipy.special import binom
from sympy.functions.combinatorial.numbers import stirling


class Tree:
    """
    This Tree class implements a binary tree object in a recursive manner, i.e. its 'left_subtree' and 'right_subtree' attributes are other Tree objects.

    The class implements the '__eq__' operator to be able to compare other trees. It returns true if both trees are non-equivalent, i.e. it does not matter which subtree is the left and the right (they can be swapped).

    The class handles the number of leaves of the whole tree in the property 'n_leaves', which computes it once the first time it is accessed and then stores it in the '_n_leaves' attributes (and for every subtrees of the tree as well). The number of leaves is computed via the method 'update_tree'.

    The class also implements a hash so that Tree objects can be used as key in a dictionary. The chosen hashing function here is the sum of the hash value of the two subtrees and of the total number of leaves of the tree. The hash is computed once when the function 'update_tree' is called and is then stored in the '_hash_value' property.
    """
    def __init__(self, left_subtree=None, right_subtree=None):
        """
        Args:
            left_subtree (Tree object): Other Tree object acting as the left subtree. If None, it means the present tree is a leaf.*
            right_subtree (Tree object): Other Tree object acting as the right subtree. If None, it means the present tree is a leaf.*

        *To be a valid tree, both left_subtree and right_subtree must be Tree objects or both must be None.
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
        """
        Updated the number of leaves and the hash value of the tree and all its subtrees. Is called the first time 'n_leaves' or 'hash_value' is accessed.

        If at some point one of the subtrees is modified by the user, this method should be called to count the new number of leaves and the new hash value of the tree.
        """
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

    @property
    def depth(self):
        if self.is_leaf():
            return 0
        else:
            return 1 + max(self.left_subtree.depth, self.right_subtree.depth)

    def is_leaf(self):
        """
        A leaf is a tree with no subtrees.
        """
        return self.left_subtree is None and self.right_subtree is None

    def is_stump(self):
        """
        A stump is a tree with two leaves as subtrees.
        """
        if self.is_leaf():
            return False
        return self.left_subtree.is_leaf() and self.right_subtree.is_leaf()

    def __repr__(self):
        if self.is_leaf():
            return 'Tree()'
        elif self.is_stump():
            return 'Tree(Tree(), Tree())'
        else:
            return f'Tree of depth {self.depth}'

    def __eq__(self, other):
        """
        Check if both trees share the same structure, up to equivalencies.
        """
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
