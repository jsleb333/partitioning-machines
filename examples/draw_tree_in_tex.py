"""
In this script, we show a minimal working example on how to draw a tree in tex with python2latex.
"""
from partitioning_machines import Tree, tree_to_tikz, draw_tree
import python2latex as p2l


leaf = Tree() # tree 1
stump = Tree(leaf, leaf) # tree 2
tree3 = Tree(stump, leaf)
tree4 = Tree(stump, stump)
tree5 = Tree(tree3, leaf)
tree6 = Tree(tree4, leaf)
tree7 = Tree(tree3, stump)
tree8 = Tree(tree3, tree3)
tree9 = Tree(tree4, stump)
tree10 = Tree(tree4, tree3)
tree11 = Tree(tree4, tree4)

trees = [
    leaf,
    stump,
    tree3,
    tree4,
    tree5,
    tree6,
    tree7,
    tree8,
    tree9,
    tree10,
    tree11
]

tikzpicture_object = tree_to_tikz(trees[8])

print(tikzpicture_object.build()) # Converts object to string usable in tex file

# Draw tree in LaTeX if pdflatex is available
draw_tree(trees[8])