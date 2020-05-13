"""
In this script, we show a minimal working example on how to draw a tree in tex with python2latex.
"""
from partitioning_machines import Tree, tree_struct_to_tikz, draw_tree_structure, tree_from_sklearn_decision_tree
import python2latex as p2l

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np

X, y = load_iris(return_X_y=True)
sklearn_tree = DecisionTreeClassifier()
sklearn_tree = sklearn_tree.fit(X, y)

tree = tree_from_sklearn_decision_tree(sklearn_tree)


tikzpicture_object = tree_struct_to_tikz(tree)

print(tikzpicture_object.build()) # Converts object to string usable in tex file

# Draw tree in LaTeX if pdflatex is available
draw_tree_structure(tree)
