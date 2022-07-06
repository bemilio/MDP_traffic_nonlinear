import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt

n_juncs=12
# Road_graph = nx.DiGraph(nx.complete_graph(n_juncs)) 
Road_graph = nx.DiGraph(nx.empty_graph(n_juncs)) 
# Road_graph = nx.DiGraph(nx.to_directed(nx.connected_watts_strogatz_graph(n_juncs, 2, 0.1))) #random directed graph
for i in range(n_juncs): # add self loops
    Road_graph.add_edge(i,i)
Road_graph.add_edge(0,1)
Road_graph.add_edge(1,0)
Road_graph.add_edge(0,2)
Road_graph.add_edge(0,3)
Road_graph.add_edge(1,2)
Road_graph.add_edge(2,0)
Road_graph.add_edge(2,3)
Road_graph.add_edge(2,5)
Road_graph.add_edge(3,0)
Road_graph.add_edge(3,4)
Road_graph.add_edge(5,2)
Road_graph.add_edge(5,6)
Road_graph.add_edge(5,7)
Road_graph.add_edge(6,7)
Road_graph.add_edge(7,6)
Road_graph.add_edge(7,5)
Road_graph.add_edge(8,1)
Road_graph.add_edge(1,8)
Road_graph.add_edge(9,5)
Road_graph.add_edge(9,4)
Road_graph.add_edge(4,9)
Road_graph.add_edge(5,9)
Road_graph.add_edge(10,7)
Road_graph.add_edge(10,2)
Road_graph.add_edge(7,10)
Road_graph.add_edge(11,4)
Road_graph.add_edge(11,8)
Road_graph.add_edge(4,11)

travel_time_roads={} # Linearization factor of the travel time func.
capacity_roads={} # Denominator in the travel time function
limit_roads={} # Used as maximum allowed cars in a road for the shared constraints

node_positions={} # only for drawing
index=0
capacity = 0.2
v_free_flow = 1
travel_time = 1
for node in Road_graph.nodes:
    node_positions.update({node: (index,index%2)})
    index = index+1
for edge in Road_graph.edges:
    if edge[0]==edge[1]:
        travel_time_roads.update({edge: 0})
        capacity_roads.update({edge: np.infty})
        limit_roads.update({edge: np.infty})
    else:
        travel_time_roads.update({edge: travel_time})
        capacity_roads.update({edge: capacity})
        limit_roads.update({edge: 2*capacity})
nx.set_node_attributes(Road_graph, values = node_positions, name = 'pos')

nx.set_edge_attributes(Road_graph, values = travel_time_roads, name = 'travel_time')
nx.set_edge_attributes(Road_graph, values = capacity_roads, name = 'capacity')
nx.set_edge_attributes(Road_graph, values = limit_roads, name = 'limit_roads')

f= open('test_graph.pkl', 'wb')  
pickle.dump([Road_graph, travel_time], f)
f.close
print("Graph created")
