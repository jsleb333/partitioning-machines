from partitioning_machines.tree import Tree
try:
    import python2latex as p2l
except ImportError:
    raise ImportError("The drawing of trees rely on the package python2latex. Please install it with 'pip install python2latex'.")

class VectorTree:
    def __init__(self, recursive_tree):
        self.recursive_tree = recursive_tree

        self.n_nodes = 2*recursive_tree.n_leaves-1
        self.left_subtrees = [0]*self.n_nodes
        self.right_subtrees = [0]*self.n_nodes
        self.layers = [0]*self.n_nodes
        self.positions = [0]*self.n_nodes

        self._vectorize_tree()
        self._deoverlap_tree()

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
                self._shift_tree(left_node, -overlap/2 - 1)
                self._shift_tree(right_node, overlap/2 + 1)

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


def tree_to_tikz(tree, min_node_distance=1.3, layer_distance=1.6, node_size=.6):
    tree = VectorTree(tree)

    pic = p2l.TexEnvironment('tikzpicture')
    pic.options += f"""leaf/.style={{draw, diamond, minimum width={node_size}cm, minimum height={2*node_size}cm, inner sep=0pt}}""",
    pic.options += f"""internal/.style={{draw, circle, minimum width={node_size}cm, inner sep=0pt}}""",

    for node in range(tree.n_nodes):
        style = 'leaf' if tree.node_is_leaf(node) else 'internal'
        pic += f'\\node[{style}](node{node}) at ({min_node_distance*tree.positions[node]/2:.3f}, {-layer_distance*tree.layers[node]:.3f}) {{}};'

    for node in range(tree.n_nodes):
        if not tree.node_is_leaf(node):
            left_node = tree.left_subtrees[node]
            right_node = tree.right_subtrees[node]

            if tree.node_is_leaf(left_node):
                pic += f'\\draw (node{node}) -- (node{left_node}.north);'
            else:
                pic += f'\\draw (node{node}) -- (node{left_node});'

            if tree.node_is_leaf(right_node):
                pic += f'\\draw (node{node}) -- (node{right_node}.north);'
            else:
                pic += f'\\draw (node{node}) -- (node{right_node});'

    return pic

def draw_tree(tree):
    doc = p2l.Document(str(tree).replace(' ', '_'), options=('varwidth',), doc_type='standalone', border='1cm')
    doc.add_package('tikz')
    del doc.packages['geometry']
    doc.add_to_preamble('\\usetikzlibrary{shapes}')
    doc += tree_to_tikz(tree)
    doc.build()
