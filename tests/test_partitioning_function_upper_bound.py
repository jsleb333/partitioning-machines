import numpy as np
from partitioning_machines.tree import Tree
from partitioning_machines.partitioning_function_upper_bound import PartitioningFunctionUpperBound, growth_function_upper_bound


class TestPatitioninFunctionUpperBound:
    def test_truncate_nominal_feat_dist(self):
        leaf = Tree()
        pfub = PartitioningFunctionUpperBound(leaf, 10)

        n_examples = 3
        nominal_feat_dist = [1,2,3,4,5]
        answer = [1,2,12]
        output = pfub._truncate_nominal_feat_dist(nominal_feat_dist, n_examples)
        assert answer == output

        n_examples = 7
        nominal_feat_dist = [1,2,3,4,5]
        answer = nominal_feat_dist
        output = pfub._truncate_nominal_feat_dist(nominal_feat_dist, n_examples)
        assert answer == output

        n_examples = 7
        nominal_feat_dist = [1,2,3,4,5,0,0]
        answer = nominal_feat_dist
        output = pfub._truncate_nominal_feat_dist(nominal_feat_dist, n_examples)
        assert answer[:5] == output

    def test_compute_upper_bound_leaf_rl_feat(self):
        leaf = Tree()
        pfub = PartitioningFunctionUpperBound(leaf, 10)
        assert pfub(4,1) == 1
        assert pfub(4,2) == 0

    def test_compute_upper_bound_stump_rl_feat(self):
        stump = Tree(Tree(), Tree())
        pfub = PartitioningFunctionUpperBound(stump, 10)
        assert pfub(4,1) == 1
        assert pfub(50,1) == 1
        assert pfub(1,2) == 0
        assert pfub(6,2) == 2**5-1
        assert pfub(7,2) < 2**6-1

    def test_compute_upper_bound_stump_nominal_feat(self):
        stump = Tree(Tree(), Tree())
        pfub = PartitioningFunctionUpperBound(stump, 0, nominal_feat_dist=[0, 3])
        assert pfub(4,1) == 1
        assert pfub(50,1) == 1
        assert pfub(1,2) == 0
        assert pfub(3,2) == 3
        assert pfub(4,2) == 6

    def test_compute_upper_bound_other_trees_rl_feat(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        pfub = PartitioningFunctionUpperBound(tree, 10)
        m = 16
        assert pfub(m,2) == 2**(m-1)-1
        m = 17
        assert pfub(m,2) < 2**(m-1)-1

    def test_compute_upper_bound_other_trees_nominal_feat(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, stump)
        pfub = PartitioningFunctionUpperBound(tree, 0, nominal_feat_dist=[0, 4])
        m = 6
        assert pfub(m,2) == 2**(m-1)-1

    def test_compute_bound_with_rl_and_nominal_features(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        nfd = [0,4,3,0,4]
        pfub = PartitioningFunctionUpperBound(tree, 10, nominal_feat_dist=nfd)
        m = 16
        assert pfub(m, 3)

    def test_compute_bound_with_rl_and_ordinal_features(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        ofd = [0,4,3,0,4]
        pfub = PartitioningFunctionUpperBound(tree, 10, ordinal_feat_dist=ofd)
        m = 16
        assert pfub(m, 3)

    def test_compute_bound_with_ordinal_and_nominal_features(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        ofd = [0,0,3,0,4]
        nfd = [0,3,0,5,0,0]
        pfub = PartitioningFunctionUpperBound(tree, 10, ordinal_feat_dist=ofd)
        m = 16
        assert pfub(m, 3)

    def test_tight_bound_is_python_int(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        ofd = [0,0,3,0,4]
        nfd = [0,3,0,5,0,0]
        pfub = PartitioningFunctionUpperBound(tree, 10, ordinal_feat_dist=ofd)
        m = 16
        assert isinstance(pfub(m, 3), int)

    def test_loose_bound_is_python_int(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        ofd = [0,0,3,0,4]
        nfd = [0,3,0,5,0,0]
        pfub = PartitioningFunctionUpperBound(tree, 10, ordinal_feat_dist=ofd, loose=True)
        m = 16
        assert isinstance(pfub(m, 3), int)

    def test_compute_bound_with_precomputed_tables(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        pfub = PartitioningFunctionUpperBound(tree, 10)
        pfub(16, 2)

        other_tree = Tree(tree, tree)
        pfub = PartitioningFunctionUpperBound(other_tree, 10, pre_computed_tables=pfub.pfub_table)
        assert pfub.pfub_table[tree][2, 16, 10, (0,)] == 2**(16-1)-1
        pfub(17, 2)

    def test_loose_bound(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        pfub = PartitioningFunctionUpperBound(tree, 10, loose=True)
        pfub(100, 2)
        pfub(100, 3)

    def test_log_loose_bound(self):
        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        pfub = PartitioningFunctionUpperBound(tree, 10, loose=True)
        log_pfub = PartitioningFunctionUpperBound(tree, 10, loose=True, log=True)
        assert log_pfub(1, 1) == 0
        m, c = 50, 3
        assert np.isclose(np.log(pfub(m, c)), log_pfub(m, c))
        m, c = 30, 5 # More parts than leaves
        assert log_pfub(m, c) == -np.inf

def test_growth_function_upper_bound():
    assert growth_function_upper_bound(Tree(Tree(), Tree()), n_rl_feat=10, n_classes=3)(1) == 3
    assert growth_function_upper_bound(Tree(Tree(), Tree()), n_rl_feat=10, n_classes=3)(2) == 3 + 6

def test_log_growth_function_upper_bound():
    assert np.isclose(growth_function_upper_bound(Tree(Tree(), Tree()), n_rl_feat=10, n_classes=3, log=True, loose=True)(1), np.log(3))
    assert np.isclose(growth_function_upper_bound(Tree(Tree(), Tree()), n_rl_feat=10, n_classes=3, log=True, loose=True)(2), np.log(3 + 6))
