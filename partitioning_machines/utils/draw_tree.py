from partitioning_machines.tree import Tree


class VectorTree:
    def __init__(self, recursive_tree):
        self.recursive_tree = recursive_tree

        self.left_subtrees = [0]*(2*recursive_tree.n_leaves-1)
        self.right_subtrees = [0]*(2*recursive_tree.n_leaves-1)
        self.layers = [0]*(2*recursive_tree.n_leaves-1)
        self.positions = [0]*(2*recursive_tree.n_leaves-1)

        self._vectorize_tree()

    def _vectorize_tree(self):
        layer = 0
        node_id = 0
        subtrees_in_layer = [(node_id, self.recursive_tree)]
        child_node_id = 1
        for layer in range(self.recursive_tree.depth+1):
            subtrees_in_next_layer = []
            for node_id, subtree in subtrees_in_layer:
                self.layers[node_id] = layer

                if subtree.is_leaf():
                    self.left_subtrees[node_id] = -1
                    self.right_subtrees[node_id] = -1
                else:
                    self.left_subtrees[node_id] = child_node_id
                    self.positions[child_node_id] = self.positions[node_id] - 1
                    subtrees_in_next_layer.append((child_node_id, subtree.left_subtree))
                    child_node_id += 1

                    self.right_subtrees[node_id] = child_node_id
                    self.positions[child_node_id] = self.positions[node_id] + 1
                    subtrees_in_next_layer.append((child_node_id, subtree.right_subtree))
                    child_node_id += 1

            subtrees_in_layer = subtrees_in_next_layer

    def node_is_leaf(self, node):
        return self.left_subtrees[node] == -1 and self.right_subtrees[node] == -1

    def _deoverlap_tree(self, node=0):
        if self.node_is_leaf(node):
            return
        else:
            left_node = self.left_subtrees[node]
            right_node = self.right_subtrees[node]
            self._deoverlap_tree(left_node)
            self._deoverlap_tree(right_node)
            overlap = self._find_largest_overlap(left_node, right_node)
            if overlap >= -1:
                self._shift_tree(left_node, overlap/2 - 1)
                self._shift_tree(right_node, -overlap/2 + 1)

    def _find_largest_overlap(self, left_node, right_node):
        rightest_positions = self._find_rightest_positions_by_layer(left_node)
        leftest_positions = self._find_leftest_positions_by_layer(right_node)
        overlaps = [r - l for l, r in zip(leftest_positions, rightest_positions)]
        return max(overlaps)

    def _find_rightest_positions_by_layer(self, node):
        rightest_positions_by_layer = []
        nodes_in_layer = [node]
        while nodes_in_layer:
            nodes_in_next_layer = []
            max_pos = self.positions[nodes_in_layer[0]]
            for node in nodes_in_layer:
                if self.positions[node] > max_pos:
                    max_pos = self.positions[node]

                if not self.node_is_leaf(node):
                    nodes_in_next_layer.append(self.left_subtrees[node])
                    nodes_in_next_layer.append(self.right_subtrees[node])
            rightest_positions_by_layer.append(max_pos)
            nodes_in_layer = nodes_in_next_layer

        return rightest_positions_by_layer

    def _find_leftest_positions_by_layer(self, node):
        leftest_positions_by_layer = []
        nodes_in_layer = [node]
        while nodes_in_layer:
            nodes_in_next_layer = []
            min_pos = self.positions[nodes_in_layer[0]]
            for node in nodes_in_layer:
                if self.positions[node] < min_pos:
                    min_pos = self.positions[node]

                if not self.node_is_leaf(node):
                    nodes_in_next_layer.append(self.left_subtrees[node])
                    nodes_in_next_layer.append(self.right_subtrees[node])
            leftest_positions_by_layer.append(min_pos)
            nodes_in_layer = nodes_in_next_layer

        return leftest_positions_by_layer

    def _shift_tree(self, node, shift):
        self.positions[node] += shift
        if self.node_is_leaf(node):
            return
        else:
            self._shift_tree(self.left_subtrees[node], shift)
            self._shift_tree(self.right_subtrees[node], shift)

def draw_tree(tree):
    vectorized_tree = VectorTree(tree)
    _deoverlap_tree(vectorized_tree)
