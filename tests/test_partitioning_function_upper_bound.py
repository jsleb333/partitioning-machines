from partitioning_machines import Tree, PartitioningFunctionUpperBound, growth_function_upper_bound


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

    def test_compute_bound_with_precomputed_tables(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        pfub = PartitioningFunctionUpperBound(tree, 10)
        pfub(16, 2)

        other_tree = Tree(tree, tree)
        pfub = PartitioningFunctionUpperBound(other_tree, 10, pfub.pfub_table)
        assert pfub.pfub_table[tree][2, 16, 10] == 2**(16-1)-1
        pfub(17, 2)


def test_growth_function_upper_bound():
    upper_bound = growth_function_upper_bound(Tree(Tree(), Tree()), n_features=10, n_classes=3)
    assert upper_bound(1) == 3
    assert upper_bound(2) == 3 + 6