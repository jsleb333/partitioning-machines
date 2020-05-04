from partitioning_machines import Tree
from partitioning_machines.utils.draw_tree import _vectorize_tree
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
    left_subtrees, right_subtrees, layers, orders, positions = _vectorize_tree(trees[2]).values()
    
    assert left_subtrees == [1,3,-1,-1,-1]
    assert right_subtrees == [2,4,-1,-1,-1]
    assert layers == [0,1,1,2,2]
    assert orders == [0,0,1,0,1]
    assert positions == [0,-1,1,-2,0]
