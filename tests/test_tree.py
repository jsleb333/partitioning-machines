from pytest import fixture, raises
from copy import deepcopy

from partitioning_machines import Tree


@fixture
def trees():
    trees = [Tree()]
    trees.append(Tree(deepcopy(trees[0]), deepcopy(trees[0])))
    trees.append(Tree(deepcopy(trees[1]), deepcopy(trees[0])))
    trees.append(Tree(deepcopy(trees[1]), deepcopy(trees[1])))
    trees.append(Tree(deepcopy(trees[2]), deepcopy(trees[0])))
    trees.append(Tree(deepcopy(trees[2]), deepcopy(trees[1])))
    trees.append(Tree(deepcopy(trees[3]), deepcopy(trees[0])))
    trees.append(Tree(deepcopy(trees[2]), deepcopy(trees[2])))
    trees.append(Tree(deepcopy(trees[3]), deepcopy(trees[1])))
    trees.append(Tree(deepcopy(trees[3]), deepcopy(trees[2])))
    trees.append(Tree(deepcopy(trees[3]), deepcopy(trees[3])))
    return trees

@fixture
def overlapping_trees(trees):
    for tree in trees:
        tree._init_position()

    return trees


class TestTree:
    def test_is_leaf(self, trees):
        assert trees[0].is_leaf()
        assert trees[1].left_subtree.is_leaf()
        assert trees[1].right_subtree.is_leaf()

    def test_is_stump(self, trees):
        assert trees[1].is_stump()
        assert trees[2].left_subtree.is_stump()

    def test_n_leaves(self, trees):
        assert trees[2].n_leaves == 3
        assert trees[2].left_subtree.n_leaves == 2

    def test_n_nodes(self, trees):
        assert trees[0].n_nodes == 0
        assert trees[2].n_nodes == 2
        assert trees[2].left_subtree.n_nodes == 1

    def test__eq__(self, trees):
        leaf = Tree()
        assert leaf == Tree()

        tree2 = Tree(Tree(leaf, leaf), leaf)
        tree2_mirror = Tree(leaf, Tree(leaf, leaf))
        assert tree2 == tree2_mirror

        assert trees[1] == trees[2].left_subtree

    def test__eq__wrong_type(self, trees):
        with raises(ValueError):
            trees[0] == 1

    def test_hash(self, trees):
        assert [hash(tree) for tree in trees[:7]] == [0,2,5,8,9,12,13]
        assert hash(trees[1].left_subtree) == 0
        assert hash(trees[2].left_subtree) == 2

    def test_depth(self, trees):
        assert [tree.depth for tree in trees[:7]] == [0,1,2,2,3,3,3]

    def test_repr(self, trees):
        assert [repr(tree) for tree in trees[:7]] == ['Tree()',
                                                  'Tree(Tree(), Tree())',
                                                  'Tree of depth 2',
                                                  'Tree of depth 2',
                                                  'Tree of depth 3',
                                                  'Tree of depth 3',
                                                  'Tree of depth 3']

    def test_layer(self, trees):
        assert trees[5].layer == 0
        assert trees[5].left_subtree.layer == 1
        assert trees[5].left_subtree.left_subtree.layer == 2

    def test_position(self, trees):
        assert trees[1].position == 0
        assert trees[1].left_subtree.position == -1
        assert trees[1].right_subtree.position == 1

    def test__len__(self, trees):
        assert [len(tree) for tree in trees[:7]] == [1, 3, 5, 7, 7, 9, 9]

    def test_is(self, trees):
        assert trees[1] is trees[1]
        assert trees[1] is not trees[1].left_subtree

    def test_in(self, trees):
        list_of_subtrees = [trees[2]]
        # assert trees[2] in list_of_subtrees
        assert trees[2].left_subtree not in list_of_subtrees

    def test_iter(self, trees):
        for subtree in trees[0]:
            assert subtree is trees[0]

        assert len([subtree for subtree in trees[1]]) == 3
        assert len([subtree for subtree in trees[2]]) == 5

    def test_replace_leaf_by_stump(self, trees):
        tree = Tree(Tree(), Tree())
        tree.left_subtree.replace_subtree(deepcopy(tree))
        assert tree == trees[2]
        assert [t.depth for t in tree] == [2, 1, 0, 0, 0]
        assert [t.layer for t in tree] == [0, 1, 2, 2, 1]
        assert [t.position for t in tree] == [0, -1, -2, 0, 1]

    def test_replace_stump_by_leaf(self, trees):
        tree = Tree(Tree(), Tree())
        tree.replace_subtree(Tree())
        assert tree.is_leaf()

    def test_replace_leaf_by_leaf(self, trees):
        tree = Tree(Tree(), Tree())
        tree.left_subtree.replace_subtree(Tree())
        assert tree == trees[1]

    def test_split_leaf(self, trees):
        tree = Tree()
        tree.split_leaf()
        assert tree == trees[1]

    def test_remove_subtree(self, trees):
        trees[2].left_subtree.remove_subtree()
        assert trees[2] == trees[1]

    def test_find_extremal_position_by_layer_max_mode(self, overlapping_trees):
        tree = overlapping_trees[2]
        assert tree._find_extremal_position_by_layer('max') == [0, 1, 0]
        assert tree.left_subtree._find_extremal_position_by_layer('max') == [-1, 0]
        assert tree.left_subtree.left_subtree._find_extremal_position_by_layer('max') == [-2]
        assert tree.left_subtree.right_subtree._find_extremal_position_by_layer('max') == [0]
        assert tree.right_subtree._find_extremal_position_by_layer('max') == [1]

        tree = overlapping_trees[10]
        print(tree.depth)
        assert tree._find_extremal_position_by_layer('max') == [0, 1, 2, 3]
        assert tree.left_subtree._find_extremal_position_by_layer('max') == [-1, 0, 1]
        assert tree.left_subtree.left_subtree._find_extremal_position_by_layer('max') == [-2, -1]
        assert tree.left_subtree.left_subtree.left_subtree._find_extremal_position_by_layer('max') == [-3]

    def test_find_extremal_position_by_layer_min_mode(self, overlapping_trees):
        tree = overlapping_trees[2]
        assert tree._find_extremal_position_by_layer('min') == [0, -1, -2]
        assert tree.left_subtree._find_extremal_position_by_layer('min') == [-1, -2]
        assert tree.left_subtree.left_subtree._find_extremal_position_by_layer('min') == [-2]
        assert tree.left_subtree.right_subtree._find_extremal_position_by_layer('min') == [0]
        assert tree.right_subtree._find_extremal_position_by_layer('min') == [1]

    def test_find_largest_overlap(self, overlapping_trees):
        tree = overlapping_trees[3]
        assert tree._find_largest_overlap() == 0
        assert tree.left_subtree._find_largest_overlap() == -2
        assert tree.right_subtree._find_largest_overlap() == -2

        tree = overlapping_trees[10]
        assert tree._find_largest_overlap() == 2
        assert tree.left_subtree._find_largest_overlap() == 0


    def test_shift_tree(self, overlapping_trees):
        tree = overlapping_trees[3]
        tree.left_subtree._shift_tree(-1)
        assert [t.position for t in tree] == [0, -2, -3, -1, 1, 0, 2]

        tree = overlapping_trees[10]
        tree.left_subtree._shift_tree(-2)
        tree.right_subtree._shift_tree(2)
        assert [t.position for t in tree] == [0, -3, -4, -5, -3, -2, -3, -1, 3, 2, 1, 3, 4, 3, 5]

    def test_deoverlap_position(self, overlapping_trees):
        tree = overlapping_trees[3]
        tree._deoverlap_position()
        assert [t.position for t in tree] == [0, -2, -3, -1, 2, 1, 3]

        tree = overlapping_trees[7]
        tree._deoverlap_position()
        assert [t.position for t in tree] == [0, -2, -3, -4, -2, -1, 2, 1, 0, 2, 3]

        tree = overlapping_trees[10]
        tree._deoverlap_position()
        assert [t.position for t in tree] == [0, -4, -6, -7, -5, -2, -3, -1, 4, 2, 1, 3, 6, 5, 7]
