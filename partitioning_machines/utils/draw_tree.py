from partitioning_machines.tree import Tree


class Node:
    def __init__(self, position, tree, parent=None, node_id=0):
        self.position = position
        self.tree = tree
        self.parent = parent
        self.node_id = node_id
        

def _compute_position_of_nodes(tree):
    layered_tree = _build_layered_tree(tree, layered_tree)
    _deoverlap_tree(tree)

def depth(tree):
    if tree.is_leaf():
        return 1
    else:
        return max(depth(tree.left_subtree), depth(tree.right_subtree)) + 1

def _build_layered_tree(tree):    
    node_id = 0
    position = 0
    parent = None
    layer = 0
    layered_tree = [[Node(position, tree, parent, node_id)]]
    while layer < depth(tree):
        layered_tree.append([])
        for node in layered_tree[layer]:
            if not node.tree.is_leaf():
                position = node.position - 1
                node_id += 1
                layered_tree[layer+1].append(Node(position, node.tree.left_subtree, node.tree, node_id))
                
                position = node.position + 1
                node_id += 1
                layered_tree[layer+1].append(Node(position, node.tree.right_subtree, node.tree, node_id))
        layer += 1
    
    del layered_tree[-1] # Remove empty list
    
    return layered_tree

def _deoverlap_tree(tree):
    if tree.is_leaf():
        return
    else:
        _deoverlap_tree(tree.left_subtree)
        overlap = _subtrees_overlap(tree.left_subtree, tree.right_subtree)
        if overlap >= 0:
            _shift_tree(tree.left_subtree, overlap/2-1)
            
        _deoverlap_tree(tree.right_subtree, overlap)

def _subtrees_overlap(left_subtree, right_subtree):
    return _rightest_position(left_subtree) - _leftest_position(right_subtree)

def _rightest_position(tree):
    if tree.is_leaf():
        return tree.position
    else:
        return max(_rightest_position(tree.left_subtree), _rightest_position(tree.right_subtree))

def _leftest_position(tree):
    if tree.is_leaf():
        return tree.position
    else:
        return min(_leftest_position(tree.left_subtree), _leftest_position(tree.right_subtree))

def _shift_tree(tree, shift):
    if tree.is_leaf():
        tree.position += shift
    else:
        _shift_tree(tree.left_subtree, shift)
        _shift_tree(tree.right_subtree, shift)
        