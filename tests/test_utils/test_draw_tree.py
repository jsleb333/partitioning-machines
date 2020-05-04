from partitioning_machines import Tree
from partitioning_machines.utils.draw_tree import VectorTree
from pytest import fixture

@fixture
def trees():
    leaf = Tree() # tree 1
    stump = Tree(leaf, leaf) # tree 2
    tree3 = Tree(stump, leaf)
    tree4 = Tree(stump, stump)
    tree5 = Tree(tree3, leaf)
    tree6 = Tree(tree4, leaf)
    tree7 = Tree(tree3, stump)
    tree8 = Tree(tree3, tree3)
    tree9 = Tree(tree4, stump)
    tree10 = Tree(tree4, tree3)
    tree11 = Tree(tree4, tree4)

    return [leaf,
            stump,
            tree3,
            tree4,
            tree5,
            tree6,
            tree7,
            tree8,
            tree9,
            tree10,
            tree11]


class TestVectorTree:
    def test_vectorize_tree(self, trees):
        vectorized_tree = VectorTree(trees[2])

        assert vectorized_tree.left_subtrees == [1,3,-1,-1,-1]
        assert vectorized_tree.right_subtrees == [2,4,-1,-1,-1]
        assert vectorized_tree.layers == [0,1,1,2,2]
        assert vectorized_tree.positions == [0,-1,1,-2,0]

    def test_node_is_leaf(self, trees):
        vectorized_tree = VectorTree(trees[2])
        assert vectorized_tree.node_is_leaf(2)
        assert vectorized_tree.node_is_leaf(3)
        assert vectorized_tree.node_is_leaf(4)

    def test_find_rightest_positions_by_layer(self, trees):
        vectorized_tree = VectorTree(trees[3])
        assert vectorized_tree._find_rightest_positions_by_layer(0) == [0, 1, 2]
        assert vectorized_tree._find_rightest_positions_by_layer(1) == [-1, 0]

    def test_find_leftest_positions_by_layer(self, trees):
        vectorized_tree = VectorTree(trees[3])
        assert vectorized_tree._find_leftest_positions_by_layer(0) == [0, -1, -2]
        assert vectorized_tree._find_leftest_positions_by_layer(2) == [1, 0]

    def test_find_largest_overlap(self, trees):
        vectorized_tree = VectorTree(trees[3])
        assert vectorized_tree._find_largest_overlap(1, 2) == 0
        assert vectorized_tree._find_largest_overlap(3, 4) == -2

    def test_shift_tree(self, trees):
        vectorized_tree = VectorTree(trees[3])
        vectorized_tree._shift_tree(1, -1)
        assert vectorized_tree.positions == [0,-2,1,-3,-1,0,2]

    def test_deoverlap_tree(self, trees):
        vectorized_tree = VectorTree(trees[3])
        vectorized_tree._deoverlap_tree()
        assert vectorized_tree.positions == [0,-2,2,-3,-1,1,3]

        vectorized_tree = VectorTree(trees[7])
        vectorized_tree._deoverlap_tree()
        assert vectorized_tree.positions == [0,-2,2,-3,-1,1,3,-4,-2,0,2]
