from partitioning_machines import Tree
from partitioning_machines.utils.draw_tree import depth, _build_layered_tree
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


def test_depth(trees):
    depths = [1,2,3,3] + [4]*7
    assert [depth(tree) for tree in trees] == depths

def test_build_layered_tree(trees):
    tree = trees[2]
    layered_tree = _build_layered_tree(tree)
    
    assert len(layered_tree) == 3
    
    assert len(layered_tree[0]) == 1
    assert len(layered_tree[1]) == 2
    assert len(layered_tree[2]) == 2
    
    assert layered_tree[0][0].node_id == 0
    assert layered_tree[0][0].position == 0
    assert layered_tree[0][0].parent == None
    assert layered_tree[0][0].tree == tree
    
    assert layered_tree[1][0].node_id == 1
    assert layered_tree[1][0].position == -1
    assert layered_tree[1][0].parent == tree
    assert layered_tree[1][0].tree == trees[1]

    assert layered_tree[1][1].node_id == 2
    assert layered_tree[1][1].position == 1
    assert layered_tree[1][1].parent == tree
    assert layered_tree[1][1].tree == trees[0]

    assert layered_tree[2][0].node_id == 3
    assert layered_tree[2][0].position == -2
    assert layered_tree[2][0].parent == trees[1]
    assert layered_tree[2][0].tree == trees[0]

    assert layered_tree[2][1].node_id == 4
    assert layered_tree[2][1].position == 0
    assert layered_tree[2][1].parent == trees[1]
    assert layered_tree[2][1].tree == trees[0]