import networkx as nx
import matplotlib.pyplot as plt
import torch

def plot_graph(data):
    #Data: PyGraph Data object 
    edge_index = data.edge_index.cpu()  #[2, num_edges]
    G = nx.Graph()

    #Add nodes
    num_nodes = data.x.size(0)
    G.add_nodes_from(range(num_nodes))

    #add edges
    edges = edge_index.t().tolist()  #Convert to list of [i, j]
    G.add_edges_from(edges)

    #Draw graph!!
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  #Force-directed layout
    nx.draw(G, pos, with_labels=True, node_color='grey', node_size=500, edge_color='gray')
    plt.show()
