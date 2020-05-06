from partitioning_machines import Tree
from partitioning_machines.utils.draw_tree import draw_tree, tree_to_tikz
from pytest import fixture

import os, sys


@fixture
def trees():
    trees = [Tree()]
    trees.append(Tree(trees[0], trees[0]))
    trees.append(Tree(trees[1], trees[0]))
    trees.append(Tree(trees[1], trees[1]))
    trees.append(Tree(trees[2], trees[0]))
    trees.append(Tree(trees[2], trees[1]))
    trees.append(Tree(trees[3], trees[0]))
    trees.append(Tree(trees[2], trees[2]))
    trees.append(Tree(trees[3], trees[1]))
    trees.append(Tree(trees[3], trees[2]))
    trees.append(Tree(trees[3], trees[3]))
    return trees


def test_tree_to_tikz(trees):
    pic = tree_to_tikz(trees[9])
    print(pic.build())

def test_draw_tree(trees):
    draw_tree(trees[9], show_pdf=False)
    os.remove('./Tree_of_depth_3.log')
    os.remove('./Tree_of_depth_3.tex')
    os.remove('./Tree_of_depth_3.aux')
    os.remove('./Tree_of_depth_3.pdf')
