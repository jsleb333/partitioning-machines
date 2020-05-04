from partitioning_machines import Tree
from partitioning_machines.utils.draw_tree import VectorTree, draw_tree
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

@fixture
def overlapping_vectorized_trees(trees):
    actual_init = VectorTree.__init__
    VectorTree.__init__ = lambda *args: None
    vectorized_trees = []
    for tree in trees:
        v_tree = VectorTree()
        v_tree.__init__ = actual_init
        v_tree.left_subtrees = [0]*(2*tree.n_leaves-1)
        v_tree.right_subtrees = [0]*(2*tree.n_leaves-1)
        v_tree.layers = [0]*(2*tree.n_leaves-1)
        v_tree.positions = [0]*(2*tree.n_leaves-1)
        v_tree.recursive_tree = tree
        v_tree._vectorize_tree()
        vectorized_trees.append(v_tree)
    VectorTree.__init__ = actual_init

    return vectorized_trees


class TestVectorTree:
    def test_vectorize_tree(self, overlapping_vectorized_trees):
        vectorized_tree = overlapping_vectorized_trees[2]

        assert vectorized_tree.left_subtrees == [1,3,-1,-1,-1]
        assert vectorized_tree.right_subtrees == [2,4,-1,-1,-1]
        assert vectorized_tree.layers == [0,1,1,2,2]
        assert vectorized_tree.positions == [0,-1,1,-2,0]

        vectorized_tree = overlapping_vectorized_trees[10]

        assert vectorized_tree.left_subtrees == [1,3,5,7,9,11,13] + [-1]*8
        assert vectorized_tree.right_subtrees == [2,4,6,8,10,12,14] + [-1]*8
        assert vectorized_tree.layers == [0,1,1,2,2,2,2] + [3]*8
        assert vectorized_tree.positions == [0,-1,1,-2,0,0,2,-3,-1,-1,1,-1,1,1,3]

    def test_node_is_leaf(self, overlapping_vectorized_trees):
        vectorized_tree = overlapping_vectorized_trees[2]
        assert vectorized_tree.node_is_leaf(2)
        assert vectorized_tree.node_is_leaf(3)
        assert vectorized_tree.node_is_leaf(4)

    def test_find_rightest_positions_by_layer(self, overlapping_vectorized_trees):
        vectorized_tree = overlapping_vectorized_trees[3]
        assert vectorized_tree._find_rightest_positions_by_layer(0) == [0, 1, 2]
        assert vectorized_tree._find_rightest_positions_by_layer(1) == [-1, 0]

        vectorized_tree = overlapping_vectorized_trees[10]
        assert vectorized_tree._find_rightest_positions_by_layer(0) == [0, 1, 2, 3]
        assert vectorized_tree._find_rightest_positions_by_layer(1) == [-1, 0, 1]
        assert vectorized_tree._find_rightest_positions_by_layer(2) == [1, 2, 3]
        assert vectorized_tree._find_rightest_positions_by_layer(3) == [-2, -1]
        assert vectorized_tree._find_rightest_positions_by_layer(4) == [0, 1]

    def test_find_leftest_positions_by_layer(self, overlapping_vectorized_trees):
        vectorized_tree = overlapping_vectorized_trees[3]
        assert vectorized_tree._find_leftest_positions_by_layer(0) == [0, -1, -2]
        assert vectorized_tree._find_leftest_positions_by_layer(2) == [1, 0]

        vectorized_tree = overlapping_vectorized_trees[10]
        assert vectorized_tree._find_leftest_positions_by_layer(0) == [0, -1, -2, -3]
        assert vectorized_tree._find_leftest_positions_by_layer(1) == [-1, -2, -3]
        assert vectorized_tree._find_leftest_positions_by_layer(2) == [1, 0, -1]
        assert vectorized_tree._find_leftest_positions_by_layer(3) == [-2, -3]
        assert vectorized_tree._find_leftest_positions_by_layer(4) == [0, -1]

    def test_find_largest_overlap(self, overlapping_vectorized_trees):
        vectorized_tree = overlapping_vectorized_trees[3]
        assert vectorized_tree._find_largest_overlap(1, 2) == 0
        assert vectorized_tree._find_largest_overlap(3, 4) == -2

        vectorized_tree = overlapping_vectorized_trees[10]
        assert vectorized_tree._find_largest_overlap(1, 2) == 2
        assert vectorized_tree._find_largest_overlap(3, 4) == 0
        assert vectorized_tree._find_largest_overlap(5, 6) == 0


    def test_shift_tree(self, overlapping_vectorized_trees):
        vectorized_tree = overlapping_vectorized_trees[3]
        vectorized_tree._shift_tree(1, -1)
        assert vectorized_tree.positions == [0,-2,1,-3,-1,0,2]

        vectorized_tree = overlapping_vectorized_trees[10]
        vectorized_tree._shift_tree(1, -2)
        vectorized_tree._shift_tree(2, 2)
        assert vectorized_tree.positions == [0,-3,3,-4,-2,2,4,-5,-3,-3,-1,1,3,3,5]

    def test_deoverlap_tree(self, overlapping_vectorized_trees):
        vectorized_tree = overlapping_vectorized_trees[3]
        vectorized_tree._deoverlap_tree()
        assert vectorized_tree.positions == [0,-2,2,-3,-1,1,3]

        vectorized_tree = overlapping_vectorized_trees[7]
        vectorized_tree._deoverlap_tree()
        assert vectorized_tree.positions == [0,-2,2,-3,-1,1,3,-4,-2,0,2]

        vectorized_tree = overlapping_vectorized_trees[10]
        vectorized_tree._deoverlap_tree()
        assert vectorized_tree.positions == [0,-4,4,-6,-2,2,6,-7,-5,-3,-1,1,3,5,7]

def test_draw_tree(trees):
    pass
