from pytest import fixture, raises

from partitioning_machines import Tree


@fixture
def trees():
    trees = [Tree()]
    trees.append(Tree(trees[0], trees[0]))
    trees.append(Tree(trees[1], trees[0]))
    trees.append(Tree(trees[1], trees[1]))
    trees.append(Tree(trees[2], trees[0]))
    trees.append(Tree(trees[2], trees[1]))
    trees.append(Tree(trees[3], trees[0]))
    return trees


class TestTree:
    def test_children(self, trees):
        assert trees[0].left_child == -1
        assert trees[0].right_child == -1

    def test_subtrees(self, trees):
        assert trees[3].current_node == 0
        assert trees[3].left_subtree.current_node == 1
        assert trees[3].right_subtree.current_node == 4
        assert trees[3].left_subtree.left_subtree.current_node == 2

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

    def test___eq__(self, trees):
        leaf = Tree()
        assert leaf == Tree()

        tree2 = Tree(Tree(leaf, leaf), leaf)
        tree2_mirror = Tree(leaf, Tree(leaf, leaf))
        assert tree2 == tree2_mirror

        assert trees[1] == trees[2].left_subtree

    def test___eq__wrong_type(self, trees):
        with raises(ValueError):
            trees[0] == 1

    def test_hash(self, trees):
        assert [hash(tree) for tree in trees] == [0,2,5,8,9,12,13]
        assert hash(trees[1].left_subtree) == 0
        assert hash(trees[2].left_subtree) == 2

    def test_depth(self, trees):
        assert [tree.depth for tree in trees] == [0,1,2,2,3,3,3]

    def test_repr(self, trees):
        assert [repr(tree) for tree in trees] == ['Tree()',
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
        assert [len(tree) for tree in trees] == [1, 3, 5, 7, 7, 9, 9]

    def test_children_list(self, trees):
        assert trees[3]._left_children == [1,2,-1,-1,5,-1,-1]
        assert trees[3]._right_children == [4,3,-1,-1,6,-1,-1]

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

        assert [subtree.current_node for subtree in trees[1]] == [0,1,2]
        assert [subtree.current_node for subtree in trees[2]] == [0,1,2,3,4]
