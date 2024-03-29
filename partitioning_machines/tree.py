"""Implementation of a binary tree"""
from copy import copy, deepcopy
from re import sub


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
                 parent=None):
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

        self.height = 0
        self.depth = 0
        self.n_leaves = 1
        self.n_nodes = 0
        self.hash_value = 0

        self.update_tree()

    @property
    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root

    def update_tree(self):
        root = self.root
        root._update_height()
        root._update_depth()
        root._update_n_leaves()
        root._update_n_nodes()
        root._update_hash_value()

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

        if self.hash_value != other.hash_value:
            return False

        if self.n_leaves != other.n_leaves:
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
        yield from self.traverse()

    def traverse(self, order='pre'):
        """
        Iterates on every subtrees of the tree in the specified order. Valid orders are 'pre', 'in' and 'post'.
        """
        if order == 'pre':
            yield self
            if not self.is_leaf():
                yield from self.left_subtree
                yield from self.right_subtree
        elif order == 'in':
            if not self.is_leaf():
                yield from self.left_subtree
            yield self
            if not self.is_leaf():
                yield from self.right_subtree
        elif order == 'post':
            if not self.is_leaf():
                yield from self.left_subtree
                yield from self.right_subtree
            yield self
        else:
            raise ValueError('Unknown traversal order.')

    def __contains__(self, obj):
        if obj is self:
            return True
        if not self.is_leaf():
            return obj in self.left_subtree or obj in self.right_subtree
        else:
            return False

    def __copy__(self):
        return self._copy(deep=False)

    def __deepcopy__(self, memo):
        return self._copy(memo=memo, deep=True)

    def _copy(self, memo=None, deep=False):
        copy_fn = deepcopy if deep else copy

        # Shallow copy to have access to references
        copy_of_dict = copy(self.__dict__)

        # Delete critical references to avoid recursion loops
        copy_of_dict['parent'] = None
        copy_of_dict['left_subtree'] = None
        copy_of_dict['right_subtree'] = None

        # Deep copy of other references if necessary
        if deep:
            copy_of_dict = deepcopy(copy_of_dict, memo)

        # Creating the new instance
        copy_of_tree = type(self).__new__(type(self))
        copy_of_tree.__dict__.update(copy_of_dict)
        if not self.is_leaf():
            left_subtree = copy_fn(self.left_subtree)
            left_subtree.parent = copy_of_tree
            copy_of_tree.left_subtree = left_subtree

            right_subtree = copy_fn(self.right_subtree)
            right_subtree.parent = copy_of_tree
            copy_of_tree.right_subtree = right_subtree

        return copy_of_tree

    def replace_subtree(self, tree, update_tree=True, inplace: bool = True):
        """
        Replaces current subtree with given tree instead.

        Args:
            tree (Tree):
                Tree object to replace current subtree.
            update_tree (bool):
                If True, will automatically update the attributes of the tree (height, lenght, n_leaves, etc.).
            inplace (bool):
                If True, will replace the subtree of the current tree. Otherwise, the current tree is (shallow) copied before the subtree is removed.

        Returns self or a new tree.
        """
        if inplace:
            subtree = self
        else:
            copy_of_tree = copy(self.root)
            subtree = copy_of_tree.follow_path(self.path_from_root())

        if subtree.parent is None: # Changing the whole tree
            subtree.__dict__ = tree.__dict__
            subtree.parent = None
        else:
            tree.parent = subtree.parent
            if subtree is subtree.parent.left_subtree:
                subtree.parent.left_subtree = tree
            else:
                subtree.parent.right_subtree = tree
            if update_tree:
                subtree.update_tree()
        return subtree

    def split_leaf(self, update_tree=True):
        """
        Makes a leaf into a node with two leaves as children.

        Returns self.
        """
        if not self.is_leaf():
            raise RuntimeError('Cannot split internal node.')
        return self.replace_subtree(Tree(Tree(), Tree()), update_tree=update_tree)

    def remove_subtree(self, update_tree: bool = True, inplace: bool = True):
        """
        Transforms the subtree into a leaf.

        Args:
            update_tree (bool):
                If True, will automatically update the attributes of the tree (height, lenght, n_leaves, etc.).
            inplace (bool):
                If True, will remove the subtree of the current tree. Otherwise, the current tree is (shallow) copied before the subtree is removed.

        Returns self or a new tree.
        """
        if inplace:
            subtree = self
        else:
            copy_of_tree = copy(self.root)
            subtree = copy_of_tree.follow_path(self.path_from_root())

        subtree.left_subtree = None
        subtree.right_subtree = None
        if update_tree:
            subtree.update_tree()
        return subtree

    def path_from_root(self):
        """
        Returns a list of strings indicating to go 'left' or 'right' for each node starting from the root node.
        """
        path = []
        parent = self.parent
        child = self
        while parent is not None:
            if child is parent.left_subtree:
                path.append('left')
            else:
                path.append('right')
            child = parent
            parent = parent.parent

        path.reverse()
        return path

    def follow_path(self, path):
        """
        Given a path (list of 'left' and 'right' strings), returns the corresponding node.
        """
        subtree = self
        for indication in path:
            if indication == 'left':
                subtree = subtree.left_subtree
            else:
                subtree = subtree.right_subtree

        return subtree
