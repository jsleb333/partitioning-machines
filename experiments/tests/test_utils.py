import sys, os
sys.path.append(os.getcwd())

from partitioning_machines import Tree
from experiments.utils import *


def test_count_node_not_stump():
    leaf = Tree()
    assert count_nodes_not_stump(leaf) == 0

    stump = Tree(leaf, leaf)
    assert count_nodes_not_stump(stump) == 0

    tree1 = Tree(stump, leaf)
    assert count_nodes_not_stump(tree1) == 1

    tree2 = Tree(stump, stump)
    assert count_nodes_not_stump(tree2) == 1

    tree3 = Tree(tree1, stump)
    assert count_nodes_not_stump(tree3) == 2
