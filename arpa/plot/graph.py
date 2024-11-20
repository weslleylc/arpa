from random import random
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_pydot import graphviz_layout


def draw_tree(TT, root=0, figsize=(30, 12), decimals=3, node_list=None, path=None, mapping=None, show_labels=False):
    T = TT.copy()
    if mapping is not None:
        T = nx.relabel_nodes(T, mapping)
        # root = root + 1
        # if node_list is not None:
        #     node_list = [n + 1 for n in node_list]

    plt.figure(figsize=figsize)
    pos = hierarchy_pos(T, root, width=1., vert_gap=0.2,
                        vert_loc=0, xcenter=0.5)
    pos_label = pos.copy()
    for k, v in pos_label.items():
        pos_label[k] = (v[0] + 0.01, v[1] + 0.01)
    if node_list is not None:
        color_map = []
        for node in T:
            if node in node_list:
                color_map.append('red')
            else:
                color_map.append('white')
        nx.draw(T, pos=pos, with_labels=True, node_color=color_map,
                edgecolors='black', linewidths=0.5)
    else:
        nx.draw(T, pos=pos, with_labels=True, node_color='white',
                edgecolors='black', linewidths=0.5)
    if show_labels:
        node_labels = nx.get_node_attributes(T, 'iteration')
        nx.draw_networkx_labels(T, pos_label, node_labels)

    edge_labels = nx.get_edge_attributes(T, 'weight')
    for key in edge_labels.keys():
        edge_labels[key] = round(edge_labels[key], decimals)
    nx.draw_networkx_edge_labels(T, pos, edge_labels)
    if path is not None:
        plt.savefig(path + ".png")
    plt.show()


def draw_components(GG, round, figsize=(10, 8), mapping=None, path=None):
    G = GG.copy()
    if mapping is not None:
        G = nx.relabel_nodes(G, mapping)
    # plt.figure(figsize=figsize)
    # plt.title(f"Aggregated features round {round}.")
    pos = graphviz_layout(G, prog="neato")
    for c in nx.connected_components(G):
        subgraph = G.subgraph(c)
        node_labels = nx.get_node_attributes(subgraph, 'weight')
        root, value = max(node_labels.items(), key=lambda x: x[1])
        color_map = []
        for node in subgraph:
            if node == root:
                color_map.append('red')
            else:
                color_map.append('white')
        nx.draw(subgraph, pos, with_labels=True)
        nodes = nx.draw_networkx_nodes(subgraph, pos, node_color=color_map)
        nx.draw_networkx_labels(subgraph, pos)
        nodes.set_edgecolor('black')
    if path is not None:
        # path = f"./figures/aggregated_features_round_{round}.png"
        plt.savefig(f"{path} + round_{round}.png")
        plt.savefig(path)
    plt.show()
    plt.close()


def draw_tree_components(GG, figsize=(10, 8), mapping=None):
    G = GG.copy()

    if mapping is not None:
        G = nx.relabel_nodes(G, mapping)
    # plt.figure(figsize=figsize)
    # plt.title(f"Aggregated features round {round}.")
    pos = graphviz_layout(G, prog="neato")
    for c in nx.connected_components(G):
        subgraph = G.subgraph(c)
        node_labels = nx.get_node_attributes(subgraph, 'cost')
        root, value = max(node_labels.items(), key=lambda x: x[1])
        draw_tree(subgraph, root=root, figsize=figsize, node_list=[root],
                  path=f"./figures/round_{round}_component_{root}")


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

