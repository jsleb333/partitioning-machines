from partitioning_machines import Tree
from partitioning_machines.utils.draw_tree import _vectorize_tree, \
    node_is_leaf, _find_rightest_positions_by_layer, _find_leftest_positions_by_layer
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

def test_vectorize_tree(trees):
    left_subtrees, right_subtrees, layers, positions = _vectorize_tree(trees[2]).values()
    
    assert left_subtrees == [1,3,-1,-1,-1]
    assert right_subtrees == [2,4,-1,-1,-1]
    assert layers == [0,1,1,2,2]
    assert positions == [0,-1,1,-2,0]

def test_node_is_leaf(trees):
    vectorized_tree = _vectorize_tree(trees[2])
    assert node_is_leaf(vectorized_tree, 2)
    assert node_is_leaf(vectorized_tree, 3)
    assert node_is_leaf(vectorized_tree, 4)

def test_find_rightest_positions_by_layer(trees):
    vectorized_tree = _vectorize_tree(trees[3])
    assert _find_rightest_positions_by_layer(vectorized_tree, 0) == [0, 1, 2]
    assert _find_rightest_positions_by_layer(vectorized_tree, 1) == [-1, 0]
    
def test_find_leftest_positions_by_layer(trees):
    vectorized_tree = _vectorize_tree(trees[3])
    assert _find_leftest_positions_by_layer(vectorized_tree, 0) == [0, -1, -2]
    assert _find_leftest_positions_by_layer(vectorized_tree, 1) == [1, 0]