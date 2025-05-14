import networkx as nx

def build_graph2():
    N = 6
    Jval = 2.0
    G = nx.Graph()
    G.add_nodes_from([i for i in range(N)])
    G.add_edges_from([(i,(i+1)% G.number_of_nodes() ) for i in range(N)])
    for e in G.edges:
        G.edges[e]['weight'] = Jval

    return G

def build_1d_graph(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(i, i+1) for i in range(n - 1)])
    for e in G.edges:
        G.edges[e]['weight'] = 1.0  # or random
    return G
