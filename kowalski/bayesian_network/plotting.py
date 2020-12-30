"""
What you'll find here
- hortizontal boxplot
- plot_graph
"""

import seaborn as sns


def horizontal_boxplot(data, xlabel:str=None, title:str=None):
    """
    columns in data correspond to a box
    index is ignored I think
    """

    ax = sns.boxplot(data=data, orient='h', palette='GnBu', whis=.9)
    # ax = sns.swarmplot(data=res30, orient='h', palette='magma')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    # ax.figure.savefig(join(save_dir, 'boxplot.png'), dpi=500, bbox_inches='tight')
    return ax


from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE


graph_attributes = {
    'splines': 'spline',  # I use splies so that we have no overlap
    'ordering': 'out',
    'ratio': 'fill',  # This is necessary to control the size of the image
    # 'size': '16,9!',  # Set the size of the final image. (this is a typical presentation size)
    'fontcolor': '#FFFFFFD9',
    # 'fontname': 'Helvetica',
    # 'fontsize': 100,
    'labeljust': 'l',
    'labelloc': 't',
    'pad': '1,1',
    'dpi': 200,
    'nodesep': 0.8,
    'ranksep': '.5 equally',
}



def _make_node_attributes(g):
    # Making all nodes hexagonal with black coloring
    node_attributes = {
        node: {
            'shape': 'hexagon',
            'width': 2.2,
            'height': 2,
            'fillcolor': '#000000',
            'penwidth': '10',
            'color': '#4a90e2d9',
            'fontsize': 35,
            'labelloc': 'c',
        }
        for node in g.nodes
    }
    return node_attributes



def _make_edge_attributes(g):
    # Customising edges
    edge_attributes = {
        (u, v): {
            'penwidth': w * 20 + 2,  # Setting edge thickness
            # 'weight': int(5 * w),  # Higher 'weight's mean shorter edges
            'arrowsize': 2 - 2.0 * w,  # Avoid too large arrows
            'arrowtail': 'dot',
        }
        for u, v, w in g.edges(data='weight')
    }
    return edge_attributes



def plot_graph(structural_model, layout='dot', rename_node_dict=None):
    """
    Plot structural model graph with some default settings.
    rename:True -> whether to rename the nodes with upper case and spacing
    """

    # edge_attr = _make_edge_attributes(structural_model)
    node_attr = _make_node_attributes(structural_model)

    for node in structural_model.nodes:
        node_attr[node]['label'] = node.replace('_', '\n').title()
        if 'Response' in node or 'response' in node:
            node_attr[node]['fillcolor'] = '#DF5F00'

    if rename_node_dict is not None:
        assert type(rename_node_dict) == dict
        for node, new_name in rename_node_dict.items():
            node_attr[node]['label'] = new_name

    viz = plot_structure(
        structural_model,
        prog=layout,
        graph_attributes=graph_attributes,
        node_attributes=node_attr,
        # edge_attributes=edge_attr

    )
    f = viz.draw(format='png')
    return Image(f)
