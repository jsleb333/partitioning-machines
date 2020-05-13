import os, sys
import numpy as np
from pytest import fixture

from partitioning_machines import Tree, DecisionTreeClassifier, gini_impurity_criterion
from partitioning_machines.utils.draw_tree import *


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


def test_tree_struct_to_tikz(trees):
    pic = tree_struct_to_tikz(trees[9])
    print(pic.build())

def test_decision_tree_to_tikz(trees):
    X = np.array([[1,2,3,4],
                  [3,4,7,3],
                  [6,7,3,2],
                  [5,5,2,6],
                  [9,1,9,5]
                  ])
    y = np.array([0,1,0,2,2])
    dtc = DecisionTreeClassifier(gini_impurity_criterion)
    dtc.fit(X, y)
    pic = decision_tree_to_tikz(dtc)
    print(pic.build())

def test_draw_tree_structure(trees):
    draw_tree_structure(trees[9], show_pdf=False)
    os.remove('./Tree_of_height_3.log')
    os.remove('./Tree_of_height_3.tex')
    os.remove('./Tree_of_height_3.aux')
    os.remove('./Tree_of_height_3.pdf')

def test_draw_decision_tree():
    X = np.array([[1,2,3,4],
                  [3,4,7,3],
                  [6,7,3,2],
                  [5,5,2,6],
                  [9,1,9,5]
                  ])
    y = ['Class 0', 'Class 1', 'Class 0', 'Class 2', 'Class 2']
    dtc = DecisionTreeClassifier(gini_impurity_criterion)
    dtc.fit(X, y)
    draw_decision_tree(dtc, show_pdf=False)
    os.remove('./Tree_of_height_2.log')
    os.remove('./Tree_of_height_2.tex')
    os.remove('./Tree_of_height_2.aux')
    os.remove('./Tree_of_height_2.pdf')
