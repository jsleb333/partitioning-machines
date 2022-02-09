import numpy as np
from copy import copy
from partitioning_machines.tree import Tree
from partitioning_machines.partitioning_function_upper_bound import PartitioningFunctionUpperBound, growth_function_upper_bound


class InspectDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_history = {}
        self.set_history = {}

    def __getitem__(self, key):
        self.get_history.setdefault(key, 0)
        self.get_history[key] += 1
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self.set_history.setdefault(key, 0)
        self.set_history[key] += 1
        return super().__setitem__(key, value)

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

    def test_precomputed_tables_are_used(self):
        table = InspectDict()

        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree = Tree(stump, leaf)
        pfub = PartitioningFunctionUpperBound(tree, 10, pre_computed_tables=table)
        pfub(16, 2)
        assert leaf not in table
        assert all(v == 1 for v in table.set_history.values())

        get_value = table.get_history[tree]
        pfub(16, 2)
        assert get_value < table.get_history[tree]

    def test_pruning_subtree_have_no_effect_on_table(self):
        table = {}

        leaf = Tree()
        stump = Tree(leaf, leaf)
        tree1 = Tree(stump, leaf)
        tree2 = Tree(stump, tree1)
        pfub = PartitioningFunctionUpperBound(tree2, 10, pre_computed_tables=table)
        pfub(16, 2)
        tree2_copy = copy(tree2)
        assert tree2_copy in table
        assert tree2_copy == list(table)[0]
        assert tree2_copy is not list(table)[0]
        table_copy = copy(table)

        tree2_copy.left_subtree.remove_subtree(inplace=True)
        assert tree2_copy != tree2
        assert tree2 in table
        assert tree2_copy not in table
        assert table_copy == table

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
