from partitioning_machines.tree import Tree


def tree_from_sklearn_decision_tree(sklearn_tree):
    """
    Returns the Tree object corresponding to the scikit-learn DecisionTreeClassifier class.

    Args:
        sklearn_tree (DecisionTreeClassifier object): Learned tree needed to be converted.
    """
    return _build_tree_from_sklearn_tree(sklearn_tree.tree_)

def _build_tree_from_sklearn_tree(sklearn_tree, current_node=0):
    children_left, children_right = sklearn_tree.children_left, sklearn_tree.children_right
    if children_left[current_node] == -1 and children_right[current_node] == -1:
        leaf = Tree()
        leaf.label = sklearn_tree.value
        return leaf
    else:
        subtree = Tree(
            _build_tree_from_sklearn_tree(sklearn_tree, children_left[current_node]),
            _build_tree_from_sklearn_tree(sklearn_tree, children_right[current_node])
        )
        subtree.rule_feature = sklearn_tree.feature
        subtree.rule_threshold = sklearn_tree.threshold
        return subtree
