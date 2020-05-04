from partitioning_machines import Tree


class TestTree:
    def test_is_leaf(self):
        leaf = Tree()
        assert leaf.is_leaf()

    def test_is_stump(self):
        stump = Tree(Tree(), Tree())
        assert stump.is_stump()

    def test_n_leaves(self):
        leaf = Tree()
        tree = Tree(Tree(leaf, leaf), leaf)
        assert tree.n_leaves == 3

    def test___eq__(self):
        leaf = Tree()
        assert leaf == Tree()

        tree1 = Tree(Tree(leaf, leaf), leaf)
        tree2 = Tree(leaf, Tree(leaf, leaf))
        assert tree1 == tree2

    def test_hash(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree1 = Tree(stump, leaf)
        tree2 = Tree(stump, stump)
        tree3 = Tree(tree1, leaf)
        tree4 = Tree(tree1, stump)
        tree5 = Tree(tree2, leaf)

        assert hash(leaf) == 0
        assert hash(stump) == 2
        assert hash(tree1) == 5
        assert hash(tree2) == 8
        assert hash(tree3) == 9
        assert hash(tree4) == 12
        assert hash(tree5) == 13

    def test_repr(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree1 = Tree(stump, leaf)
        tree2 = Tree(stump, stump)
        tree3 = Tree(tree1, leaf)
        tree4 = Tree(tree1, stump)
        tree5 = Tree(tree2, leaf)

        assert leaf.depth == 0
        assert stump.depth == 1
        assert tree1.depth == 2
        assert tree2.depth == 2
        assert tree3.depth == 3
        assert tree4.depth == 3
        assert tree5.depth == 3

    def test_repr(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree1 = Tree(stump, leaf)
        tree2 = Tree(stump, stump)
        tree3 = Tree(tree1, leaf)
        tree4 = Tree(tree1, stump)
        tree5 = Tree(tree2, leaf)

        assert repr(leaf) == 'Tree()'
        assert repr(stump) == 'Tree(Tree(), Tree())'
        assert repr(tree1) == 'Tree of depth 2'
        assert repr(tree2) == 'Tree of depth 2'
        assert repr(tree3) == 'Tree of depth 3'
        assert repr(tree4) == 'Tree of depth 3'
        assert repr(tree5) == 'Tree of depth 3'
        