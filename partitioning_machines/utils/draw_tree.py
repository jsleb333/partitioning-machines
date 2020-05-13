from partitioning_machines.tree import Tree
try:
    import python2latex as p2l
except ImportError:
    raise ImportError("The drawing of trees rely on the package python2latex. Please install it with 'pip install python2latex'.")


def tree_struct_to_tikz(tree, min_node_distance=1.3, depth_distance=1.6, node_size=.6):

    pic = p2l.TexEnvironment('tikzpicture')
    pic.options += f"""leaf/.style={{draw, diamond, minimum width={node_size}cm, minimum height={2*node_size}cm, inner sep=0pt}}""",
    pic.options += f"""internal/.style={{draw, circle, minimum width={node_size}cm, inner sep=0pt}}""",

    for node, subtree in enumerate(tree):
        style = 'leaf' if subtree.is_leaf() else 'internal'
        pic += f'\\node[{style}](node{node}) at ({min_node_distance*subtree.position/2:.3f}, {-depth_distance*subtree.depth:.3f}) {{}};'
        subtree.node_id = node

    for subtree in tree:
        if not subtree.is_leaf():

            if subtree.left_subtree.is_leaf():
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.left_subtree.node_id}.north);'
            else:
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.left_subtree.node_id});'

            if subtree.right_subtree.is_leaf():
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.right_subtree.node_id}.north);'
            else:
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.right_subtree.node_id});'

    return pic

def draw_tree(tree, show_pdf=True):
    doc = p2l.Document(str(tree).replace(' ', '_'), options=('varwidth',), doc_type='standalone', border='1cm')
    doc.add_package('tikz')
    del doc.packages['geometry']
    doc.add_to_preamble('\\usetikzlibrary{shapes}')
    doc += tree_struct_to_tikz(tree)
    doc.build(show_pdf=show_pdf)
