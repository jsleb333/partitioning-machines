"""Implementation of a binary tree"""
from copy import copy, deepcopy


class Tree:
    """
    This Tree class implements a binary tree object in a recursive fashion. The left and right subtrees are thus Tree objects too.

    Attributes maintained by the class are the number of leaves and of internal nodes of the tree, the height of the tree, the depth of the tree (relative to its parent tree), the position of the current node (relative to its parent tree) to be able to draw the tree, and a hash value to be able to hash a tree in a dictionnary. The tree class computes automatically all these quantities at the initialization and whenever the tree is modified via the provided methods to do so.

    The API also provides utilitary methods to handle the tree, such as 'is_leaf', 'is_stump', 'replace_subtree', 'split_leaf' and 'remove_subtree'.

    It also implements the '__eq__' operator to be able to compare other trees. It returns True if both tree *structures* are equivalent, i.e. it does not matter which subtree is the left and the right (they can be swapped), and neither the content of the nodes does. The '__len__' operator returns the total number of nodes in the tree. The '__iter__' operator iterates in pre-order on the subtrees of the tree.
    """
    def __init__(self,
                 left_subtree=None,
                 right_subtree=None,
                 parent=None,
                 height=0,
                 depth=0,
                 n_leaves=1,
                 n_nodes=0,
                 hash_value=0,
                 position=0):
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
        if left_subtree is not None:
            self.left_subtree.parent = self
        if right_subtree is not None:
            self.right_subtree.parent = self
        
        self.parent = parent
        
        self.height = height
        self.depth = depth
        self.n_leaves = n_leaves
        self.n_nodes = n_nodes
        self.hash_value = hash_value
        self.position = position

        self.update_tree()
    
    @property
    def tree_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.tree_root

    def update_tree(self):
        self.tree_root._update_height()
        self.tree_root._update_depth()
        self.tree_root._update_n_leaves()
        self.tree_root._update_n_nodes()
        self.tree_root._update_hash_value()
        self.tree_root._update_position()
    
    def is_leaf(self):
        return self.left_subtree is None and self.right_subtree is None

    def is_stump(self):
        """
        A stump is a tree with two leaves as subtrees.
        """
        if self.is_leaf():
            return False
        return self.left_subtree.is_leaf() and self.right_subtree.is_leaf()
    
    def _update_height(self):
        if self.is_leaf():
            self.height = 0
        else:
            self.left_subtree._update_height()
            self.right_subtree._update_height()
            self.height = 1 + max(self.left_subtree.height, self.right_subtree.height)

    def _update_depth(self, depth=0):
        self.depth = depth
        if not self.is_leaf():
            self.left_subtree._update_depth(depth+1)
            self.right_subtree._update_depth(depth+1)

    def _update_n_leaves(self):
        if self.is_leaf():
            self.n_leaves = 1
        else:
            self.left_subtree._update_n_leaves()
            self.right_subtree._update_n_leaves()
            self.n_leaves = self.left_subtree.n_leaves + self.right_subtree.n_leaves

    def _update_n_nodes(self):
        if self.is_leaf():
            self.n_nodes = 0
        else:
            self.left_subtree._update_n_nodes()
            self.right_subtree._update_n_nodes()
            self.n_nodes = 1 + self.left_subtree.n_nodes + self.right_subtree.n_nodes

    def _update_hash_value(self):
        if self.is_leaf():
            self.hash_value = 0
        else:
            self.left_subtree._update_hash_value()
            self.right_subtree._update_hash_value()
            self.hash_value = self.n_leaves + self.left_subtree.hash_value + self.right_subtree.hash_value

    def _update_position(self):
        self.tree_root._init_position()
        self.tree_root._deoverlap_position()

    def _init_position(self, position=0):
        self.position = position
        if not self.is_leaf():
            self.left_subtree._init_position(position - 1)
            self.right_subtree._init_position(position + 1)

    def _deoverlap_position(self):
        if self.is_leaf():
            return
        else:
            self.left_subtree._deoverlap_position()
            self.right_subtree._deoverlap_position()
            overlap = self._find_largest_overlap()
            if overlap >= -1:
                self.left_subtree._shift_tree(-overlap/2 - 1)
                self.right_subtree._shift_tree(overlap/2 + 1)

    def _find_largest_overlap(self):
        rightest_position = self.left_subtree._find_extremal_position_by_depth('max')
        leftest_position = self.right_subtree._find_extremal_position_by_depth('min')
        overlaps = [r - l for l, r in zip(leftest_position, rightest_position)]
        return max(overlaps)

    def _find_extremal_position_by_depth(self, mode):
        extremal_position_by_depth = []
        subtrees_in_depth = [self]
        while subtrees_in_depth:
            subtrees_in_next_depth = []
            extremal_pos = subtrees_in_depth[0].position
            for subtree in subtrees_in_depth:
                if mode == 'max':
                    if subtree.position > extremal_pos:
                        extremal_pos = subtree.position
                elif mode == 'min':
                    if subtree.position < extremal_pos:
                        extremal_pos = subtree.position
                if not subtree.is_leaf():
                    subtrees_in_next_depth.append(subtree.left_subtree)
                    subtrees_in_next_depth.append(subtree.right_subtree)
            extremal_position_by_depth.append(extremal_pos)
            subtrees_in_depth = subtrees_in_next_depth

        return extremal_position_by_depth

    def _shift_tree(self, shift):
        self.position += shift
        if not self.is_leaf():
            self.left_subtree._shift_tree(shift)
            self.right_subtree._shift_tree(shift)

    def __repr__(self):
        if self.is_leaf():
            return 'Tree()'
        elif self.is_stump():
            return 'Tree(Tree(), Tree())'
        else:
            return f'Tree of height {self.height}'

    def __eq__(self, other):
        """
        Check if both trees share the same structure, up to equivalencies.
        """
        if not isinstance(other, Tree):
            raise ValueError('Cannot compare objects.')

        if self.is_leaf():
            if other.is_leaf():
                return True
            else:
                return False

        if other.is_leaf():
            return False

        if (self.left_subtree == other.left_subtree and self.right_subtree == other.right_subtree) \
            or (self.left_subtree == other.right_subtree and self.right_subtree == other.left_subtree):
            return True

    def __hash__(self):
        # The hash is the sum of the heights d_i of each leaf i.
        return self.hash_value

    def __len__(self):
        return self.n_leaves + self.n_nodes

    def __iter__(self):
        """
        Iterates on every subtrees of the tree in a pre-order fashion.
        """
        yield self
        if not self.is_leaf():
            yield from self.left_subtree
            yield from self.right_subtree
        
    def __deepcopy__(self, memo):
        shallow_copy_of_tree = copy(self)
        
        # Get rid of critical attributes without affecting the original tree
        del shallow_copy_of_tree.left_subtree
        del shallow_copy_of_tree.right_subtree
        del shallow_copy_of_tree.parent
        
        # Create the deepcopy
        deepcopy_of_tree = type(self)(left_subtree=deepcopy(self.left_subtree),
                                      right_subtree=deepcopy(self.right_subtree))
        # Deepcopy of other non critical attributes
        deepcopy_of_tree.__dict__.update(deepcopy(shallow_copy_of_tree.__dict__))
        return deepcopy_of_tree

    def replace_subtree(self, tree):
        """
        Replaces current subtree with given tree instead.

        Returns self.
        """
        if self.parent is None: # Changing the whole tree
            self.__dict__ = tree.__dict__
        else:
            if self is self.parent.left_subtree:
                self.parent.left_subtree = tree
            else:
                self.parent.right_subtree = tree
            self.update_tree()
        return self

    def split_leaf(self):
        """
        Makes a leaf into a node with two leaves as children.

        Returns self.
        """
        if not self.is_leaf():
            raise RuntimeError('Cannot split internal node.')
        return self.replace_subtree(Tree(Tree(), Tree()))

    def remove_subtree(self):
        """
        Transforms the subtree into a leaf.

        Returns self.
        """
        return self.replace_subtree(Tree())
