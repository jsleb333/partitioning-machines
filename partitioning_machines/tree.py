"""Implementation of a binary tree"""
import numpy as np
from scipy.special import binom
from sympy.functions.combinatorial.numbers import stirling


class _TreeView:
    """
    Implements a view of a tree maintaining the current node inspected on a tree. Allows to navigate a tree easily in a recursive manner.
    """
    def __init__(self, tree, current_node=0):
        self.current_node = current_node
        self.tree = tree
    
    @property
    def left_child(self):
        return self.tree._left_children[self.current_node]

    @property
    def left_subtree(self):
        return _TreeView(self.tree, self.left_child)

    @property
    def right_child(self):
        return self.tree._right_children[self.current_node]

    @property
    def right_subtree(self):
        return _TreeView(self.tree, self.right_child)

    def __getattr__(self, name):
        if name in ['layer', 'position', 'n_leaves', 'n_nodes', 'depth', 'hash_value']:
            return getattr(self.tree, '_' + name)[self.current_node]
        else:
            return getattr(self.tree, name)

    def is_leaf(self):
        """
        A leaf is a tree with no subtrees.
        """
        return self.left_child == -1 and self.right_child == -1

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
        if not isinstance(other, _TreeView):
            raise ValueError('Cannot compare objects.')
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

    def __len__(self):
        return self.n_leaves + self.n_nodes


class Tree:
    """
    This Tree class implements a binary tree object with a set of arrays handling attributes of every nodes. This type of implementation has various advantages, but is less easy to manipulate. To overcome this, a recursive implementation is simulated with the use of an internal flag 'current_node' which situates the selected node and thus the respective subtrees. The whole tree is considered when 'current_node' is equal to 0. All attributes relevant to the current node are accessible via non-underscored names, while global attributes contained in arrays are stored in variables beginning with an underscore.

    The class implements the '__eq__' operator to be able to compare other trees. It returns true if both trees are non-equivalent, i.e. it does not matter which subtree is the left and the right (they can be swapped).

    The class handles the number of leaves of the whole tree in the property 'n_leaves', which computes it once the first time it is accessed and then stores it in the '_n_leaves' attributes (and for every subtrees of the tree as well). The number of leaves is computed via the method 'update_tree'.

    The class also implements a hash so that Tree objects can be used as key in a dictionary. The chosen hashing function here is the sum of the hash value of the two subtrees and of the total number of leaves of the tree. The hash is computed once when the function 'update_tree' is called and is then stored in the '_hash_value' property.
    """
    def __new__(cls, *args, **kwargs):
        tree = super().__new__(cls)
        tree.__init__(*args, **kwargs)
        return _TreeView(tree)
    
    def __init__(self, left_subtree=None, right_subtree=None):
        """
        Args:
            left_subtree (Tree object): Other Tree object acting as the left subtree. If None, it means the present tree is a leaf.*
            right_subtree (Tree object): Other Tree object acting as the right subtree. If None, it means the present tree is a leaf.*

        *To be a valid tree, both left_subtree and right_subtree must be Tree objects or both must be None.
        """
        if left_subtree is None and right_subtree is not None or left_subtree is not None and right_subtree is None:
            raise ValueError('Both subtrees must be either None or other valid trees.')
        
        if left_subtree is None and right_subtree is None: # Tree is a leaf
            self._left_children = [-1]
            self._right_children = [-1]
            self._layer = [0]
            self._position = [0]
            self._n_leaves = [1]
            self._n_nodes = [0]
            self._depth = [0]
            self._hash_value = [0]
        else:
            
            self._build_tree_from_subtrees(left_subtree, right_subtree)
    
    def _build_tree_from_subtrees(self, left_subtree, right_subtree):
        self._left_children = [1] \
            + [1+child if child != - 1 else -1 for child in left_subtree._left_children] \
            + [1+len(left_subtree)+child if child != - 1 else -1 for child in right_subtree._left_children]
        
        self._right_children = [1+len(left_subtree)] \
            + [1+child if child != - 1 else -1 for child in left_subtree._right_children] \
            + [1+len(left_subtree)+child if child != - 1 else -1 for child in right_subtree._right_children]
        
        self._position = [0] + [-1 + pos for pos in left_subtree._position] \
                             + [1 + pos for pos in right_subtree._position]
        
        self._layer = [0] + [1 + layer for layer in left_subtree._layer] \
                          + [1 + layer for layer in right_subtree._layer]

        self._n_leaves = [left_subtree.n_leaves + right_subtree.n_leaves] \
                         + left_subtree._n_leaves + right_subtree._n_leaves

        self._n_nodes = [1 + left_subtree.n_nodes + right_subtree.n_nodes] \
                        + left_subtree._n_nodes + right_subtree._n_nodes

        self._depth = [1 + max(left_subtree.depth, right_subtree.depth)] \
                      + left_subtree._depth + right_subtree._depth
        
        self._hash_value = [self._n_leaves[0] + left_subtree.hash_value + right_subtree.hash_value]\
                           + left_subtree._hash_value + right_subtree._hash_value
