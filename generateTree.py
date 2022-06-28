import networkx as nx
from cmath import inf
def generateTree(complexity):
    n_nodes = complexity
    G = nx.generators.complete_graph(n_nodes)
    for i in G.nodes: # add self loops
        G.add_edge(i,i)
    G = nx.to_directed(G)
    travel_time_derivative_roads={}
    capacity_roads={}
    limit_roads={}
    is_city_road={}
    for edge in G.edges:
        travel_time_derivative_roads.update({edge: 1})
        capacity_roads.update({edge: inf})
        limit_roads.update({edge: inf})
        is_city_road.update({edge: True})
    nx.set_edge_attributes(G, values = travel_time_derivative_roads, name = 'nabla_travel_time')
    nx.set_edge_attributes(G, values = capacity_roads, name = 'capacity')
    nx.set_edge_attributes(G, values = limit_roads, name = 'limit_roads')
    nx.set_edge_attributes(G, values = is_city_road, name = 'is_city_road')
    return(G)