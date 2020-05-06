"""Implementation of a binary tree"""
from copy import copy


class _TreeView:
    """
    Implements a view of a tree maintaining the current node inspected on a tree. Acts as an API and allows to navigate a tree easily in a recursive manner. See the Tree class documentation.
    """
    def __init__(self, tree, current_node=0):
        self.current_node = current_node
        self._tree = tree

    @property
    def left_child(self):
        return self._tree._left_children[self.current_node]

    @property
    def left_subtree(self):
        return _TreeView(self._tree, self.left_child)

    @property
    def right_child(self):
        return self._tree._right_children[self.current_node]

    @property
    def right_subtree(self):
        return _TreeView(self._tree, self.right_child)

    def __getattr__(self, name):
        if name in ['layer', 'position', 'n_leaves', 'n_nodes', 'depth', 'hash_value']:
            return getattr(self._tree, '_' + name)[self.current_node]
        else:
            return getattr(self._tree, name)

    def is_leaf(self):
        """
        A leaf is a tree with no subtrees.
        """
        return self._tree.node_is_leaf(self.current_node)

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
        # The hash is the sum of the depths d_i of each leaf i.
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

    def replace_subtree(self, tree):
        """
        Replaces current subtree with given tree instead.

        Returns self.
        """
        self._tree._replace_subtree(self.current_node, tree._tree)
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


class Node:
    def __init__(self,
                 left_child=None,
                 right_child=None,
                 depth=0,
                 layer=0,
                 n_leaves=1,
                 n_nodes=0,
                 hash_value=0,
                 position=0):
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.layer = layer
        self.n_leaves = n_leaves
        self.n_nodes = n_nodes
        self.hash_value = hash_value
        self.position = position

    def is_leaf(self):
        return self.left_child is None and self.right_child is None


class Tree:
    """
    This Tree class implements a binary tree object with a set of arrays handling attributes of every nodes. This type of implementation has various advantages, but is less easy to manipulate. To overcome this, a recursive implementation is simulated in the _TreeView API class. Such an object is returned when Tree() is instanciated.

    This API contains a reference to the actual tree and an internal flag 'current_node' pointing to a particular node, allowing to handle the associated subtree. The root is always at 0, thus the whole tree is considered when 'current_node' is equal to 0. All attributes relevant to the current node are accessible via non-underscored names, while global attributes contained in arrays are stored in variables beginning with an underscore.

    Attributes maintained by the class are the number of leaves and of internal nodes of the subtrees, the depth of the subtrees, the layer of the current node (relative to the whole tree), the position of the current node (relative to the whole tree) to be able to draw the tree, and a hash value to be able to hash a subtree in a dictionnary. The tree class computes automatically all these quantities at the initialization and whenever the tree is modified via the provided methods to do so.

    The API also provides utilitary methods to handle the tree, such as 'left_subtree' and 'right_subtree' properties to get a _TreeView of the subtrees and methods 'is_leaf' and 'is_stump'. Moreover, the tree can be modified using the 'replace_subtree', 'split_leaf' and 'remove_subtree' methods.

    It also implements the '__eq__' operator to be able to compare other trees. It returns true if both trees are non-equivalent, i.e. it does not matter which subtree is the left and the right (they can be swapped).
    The '__len__' operator returns the total number of nodes in the subtree.
    The '__iter__' operator iterates in pre-order on the subtrees of the subtree.
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
            self.root = Node()
        else:
            self.root = Node(left_subtree.root, right_subtree.root)#] \
                        # + [copy(node) for node in left_subtree._nodes] \
                        # + [copy(node) for node in right_subtree._nodes]

        self._nodes = []

        self.update_tree()

    def __copy__(self):
        if self.root.is_leaf():
            copied_tree = Tree()
            copied_tree.root = copy(self.root)
            return copied_tree
        else:
            copy_of_left_child = copy(self.root.left_child)
            copy_of_right_child = copy(self.root.right_child)
            copied_tree.root = copy(self.root)
            copied_tree = Tree()
            copied_tree.root.
            return Tree()

    def update_tree(self):

        self.preorder(self.root)

        self._update_depth(self.root)
        self._update_layer(self.root)
        self._update_n_leaves(self.root)
        self._update_n_nodes(self.root)
        self._update_hash_value(self.root)
        self._update_position(self.root)

    def preorder(self, node):
        self._nodes.append(node)
        if not node.is_leaf():
            self.preorder(node.left_child)
            self.preorder(node.right_child)

    def _update_depth(self, node):
        if node.is_leaf():
            node.depth = 0
        else:
            left_child, right_child = node.left_child, node.right_child
            self._update_depth(left_child)
            self._update_depth(right_child)
            node.depth = 1 + max(left_child.depth, right_child.depth)

    def _update_layer(self, node, layer=0):
        node.layer = layer
        if not node.is_leaf():
            left_child, right_child = node.left_child, node.right_child
            self._update_layer(left_child, layer+1)
            self._update_layer(right_child, layer+1)

    def _update_n_leaves(self, node):
        if node.is_leaf():
            node.n_leaves = 1
        else:
            left_child, right_child = node.left_child, node.right_child
            self._update_n_leaves(left_child)
            self._update_n_leaves(right_child)
            node.n_leaves = left_child.n_leaves + right_child.n_leaves

    def _update_n_nodes(self, node):
        if node.is_leaf():
            node.n_nodes = 1
        else:
            left_child, right_child = node.left_child, node.right_child
            self._update_n_nodes(left_child)
            self._update_n_nodes(right_child)
            node.n_nodes = 1 + left_child.n_nodes + right_child.n_nodes

    def _update_hash_value(self, node):
        if node.is_leaf():
            node.hash_value = 0
        else:
            left_child, right_child = node.left_child, node.right_child
            self._update_hash_value(left_child)
            self._update_hash_value(right_child)
            node.hash_value = node.n_leaves + left_child.hash_value + right_child.hash_value

    def _update_position(self, node):
        self._init_position(node)
        self._deoverlap_position(node)

    def _init_position(self, node, position=0):
        node.position = position
        if not node.is_leaf():
            left_child, right_child = node.left_child, node.right_child
            self._init_position(left_child, position-1)
            self._init_position(right_child, position+1)

    def _deoverlap_position(self, node):
        if node.is_leaf():
            return
        else:
            left_child, right_child = node.left_child, node.right_child
            self._deoverlap_position(left_child)
            self._deoverlap_position(right_child)
            overlap = self._find_largest_overlap(left_child, right_child)
            if overlap >= -1:
                self._shift_tree(left_child, -overlap/2 - 1)
                self._shift_tree(right_child, overlap/2 + 1)

    def _find_largest_overlap(self, left_child, right_child):
        rightest_position = self._find_rightest_position_by_layer(left_child)
        leftest_position = self._find_leftest_position_by_layer(right_child)
        overlaps = [r - l for l, r in zip(leftest_position, rightest_position)]
        return max(overlaps)

    def _find_rightest_position_by_layer(self, node):
        rightest_position_by_layer = []
        nodes_in_layer = [node]
        while nodes_in_layer:
            nodes_in_next_layer = []
            max_pos = nodes_in_layer[0].position
            for node in nodes_in_layer:
                if node.position > max_pos:
                    max_pos = node.position

                if not node.is_leaf():
                    nodes_in_next_layer.append(node.left_child)
                    nodes_in_next_layer.append(node.right_child)
            rightest_position_by_layer.append(max_pos)
            nodes_in_layer = nodes_in_next_layer

        return rightest_position_by_layer

    def _find_leftest_position_by_layer(self, node):
        leftest_position_by_layer = []
        nodes_in_layer = [node]
        while nodes_in_layer:
            nodes_in_next_layer = []
            min_pos = nodes_in_layer[0].position
            for node in nodes_in_layer:
                if node.position < min_pos:
                    min_pos = node.position

                if not node.is_leaf():
                    nodes_in_next_layer.append(node.left_child)
                    nodes_in_next_layer.append(node.right_child)
            leftest_position_by_layer.append(min_pos)
            nodes_in_layer = nodes_in_next_layer

        return leftest_position_by_layer

    def _shift_tree(self, node, shift):
        node.position += shift
        if not node.is_leaf():
            self._shift_tree(node.left_child, shift)
            self._shift_tree(node.right_child, shift)

    def _replace_subtree(self, node, tree):
        """
        Removes the subtree situated at 'node' and inserts the tree 'tree' in its place.
        """
        if tree.node_is_leaf(0):
            node.left_child = None
            node.right_child = None
        else:
            shift = len(self._left_children)-1
            self._left_children += [shift+child if child != - 1 else -1 for child in tree._left_children[1:]]
            node.left_child = shift+1

            self._right_children += [shift+child if child != - 1 else -1 for child in tree._right_children[1:]]
            node.right_child = shift+2

        self.update_tree()
