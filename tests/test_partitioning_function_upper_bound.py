from partitioning_machines import Tree, PartitioningFunctionUpperBound


class TestPatitioninFunctionUpperBound:
    def test__compute_list_of_distinct_subtrees(self):
        leaf = Tree()
        stump = Tree(Tree(), Tree())
        tree = Tree(Tree(Tree(), Tree()), Tree(Tree(), Tree()))
        pfub = PartitioningFunctionUpperBound(tree, 10)
        pfub._compute_list_of_distinct_subtrees(pfub.tree)

        assert pfub.subtrees == [leaf, stump, tree]

    def test_compute_upper_bound_leaf(self):
        leaf = Tree()
        pfub = PartitioningFunctionUpperBound(leaf, 10)
        assert pfub(4,1) == 1
        assert pfub(4,2) == 0

    def test_compute_upper_bound_stump(self):
        stump = Tree(Tree(), Tree())
        pfub = PartitioningFunctionUpperBound(stump, 10)
        assert pfub(4,1) == 1
        assert pfub(50,1) == 1
        assert pfub(1,2) == 0
        assert pfub(6,2) == 2**5-1
        assert pfub(7,2) < 2**6-1

    def test_compute_upper_bound_other_trees(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        pfub = PartitioningFunctionUpperBound(tree, 10)
        m = 16
        assert pfub(m,2) == 2**(m-1)-1
        m = 17
        assert pfub(m,2) < 2**(m-1)-1
