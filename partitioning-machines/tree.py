"""Implementation of a binary tree"""


class Tree:
    def __init__(self, left_subtree=None, right_subtree=None, parent=None):
        """
        Args:
            left_subtree ():
            right_subtree ():
            parent ():
        """
        self.left_subtree = left_subtree
        self.right_subtree = right_subtree
        self.parent = parent

    @property
    def n_leaves(self):
        if self.is_leaf():
            return 1
        else:
            return self.left_subtree.n_leaves + self.right_subtree.n_leaves

    def is_leaf(self):
        return self.left_subtree is None and self.right_subtree is None

    def is_stump(self):
        return self.left_subtree.is_leaf() and self.right_subtree.is_leaf()
