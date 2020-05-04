from partitioning_machines.tree import Tree


class Node:
    def __init__(self, position, tree, parent=None, node_id=0):
        self.position = position
        self.tree = tree
        self.parent = parent
        self.node_id = node_id
        

def _compute_position_of_nodes(tree):
    vectorized_tree = _vectorize_tree(tree)
    _deoverlap_tree(tree)
    return tree

def _vectorize_tree(tree):
    left_subtrees, right_subtrees, layers, orders, positions = [[0]*(2*tree.n_leaves-1) for i in range(5)]
    layer = 0
    node_id = 0
    subtrees_in_layer = [(node_id, tree)]
    child_node_id = 1
    for layer in range(tree.depth+1):
        subtrees_in_next_layer = []
        for node_id, subtree in subtrees_in_layer:
            layers[node_id] = layer
            
            if subtree.is_leaf():
                left_subtrees[node_id] = -1
                right_subtrees[node_id] = -1
            else:
                left_subtrees[node_id] = child_node_id
                positions[child_node_id] = positions[node_id] - 1
                subtrees_in_next_layer.append((child_node_id, subtree.left_subtree))
                child_node_id += 1
                
                right_subtrees[node_id] = child_node_id
                positions[child_node_id] = positions[node_id] + 1
                subtrees_in_next_layer.append((child_node_id, subtree.right_subtree))
                child_node_id += 1
                
        subtrees_in_layer = subtrees_in_next_layer
    
    vectorized_tree = {'left_subtrees':left_subtrees,
                       'right_subtrees':right_subtrees,
                       'layers':layers,
                       'positions':positions}
    
    return vectorized_tree

def _deoverlap_tree(tree, node=0):
    if node_is_leaf(tree, node):
        return
    else:
        left_node = tree['left_subtrees'][node]
        right_node = tree['right_subtrees'][node]
        _deoverlap_tree(tree, left_node)
        _deoverlap_tree(tree, right_node)
        overlap = _find_largest_overlap(tree, left_node, right_node)
        if overlap >= 0:
            _shift_tree(tree, left_node, -overlap/2 - 1)
            _shift_tree(tree, right_node, overlap/2 + 1)

def node_is_leaf(tree, node):
    return tree['left_subtrees'][node] == -1 and tree['right_subtrees'][node] == -1

def _find_largest_overlap(tree, left_node, right_node):
    rightest_positions = _find_rightest_positions_by_layer(tree, left_node)
    leftest_positions = _find_leftest_positions_by_layer(tree, right_node)
    overlaps = [l - r for l, r in zip(leftest_positions, rightest_positions)]
    return max(overlaps)

def _find_rightest_positions_by_layer(tree, node):
    rightest_positions_by_layer = []
    nodes_in_layer = [node]
    while nodes_in_layer:
        nodes_in_next_layer = []
        max_pos = tree['positions'][nodes_in_layer[0]]
        for node in nodes_in_layer:
            if tree['positions'][node] > max_pos:
                max_pos = tree['positions'][node]
            
            if not node_is_leaf(tree, node):
                nodes_in_next_layer.append(tree['left_subtrees'][node])
                nodes_in_next_layer.append(tree['right_subtrees'][node])
        rightest_positions_by_layer.append(max_pos)
        nodes_in_layer = nodes_in_next_layer
    
    return rightest_positions_by_layer

def _find_leftest_positions_by_layer(tree, node):
    leftest_positions_by_layer = []
    nodes_in_layer = [node]
    while nodes_in_layer:
        nodes_in_next_layer = []
        min_pos = tree['positions'][nodes_in_layer[0]]
        for node in nodes_in_layer:
            if tree['positions'][node] < min_pos:
                min_pos = tree['positions'][node]
            
            if not node_is_leaf(tree, node):
                nodes_in_next_layer.append(tree['left_subtrees'][node])
                nodes_in_next_layer.append(tree['right_subtrees'][node])
        leftest_positions_by_layer.append(min_pos)
        nodes_in_layer = nodes_in_next_layer
    
    return leftest_positions_by_layer

def _shift_tree(tree, node, shift):
    tree['positions'][node] + shift
    if node_is_leaf(node):
        return
    else:
        _shift_tree(tree, tree['left_subtrees'][node], shift)
        _shift_tree(tree, tree['right_subtrees'][node], shift)
        