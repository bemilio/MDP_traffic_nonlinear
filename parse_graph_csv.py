import csv
import networkx as nx
import numpy as np
from computeTravelTime import computeTravelTime
import pickle
import matplotlib.pyplot as plt


with open('manhattan_reduced_graph_80.csv', 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    n_juncs = 65
    travel_time_derivative_roads={} # Linearization factor of the travel time func.
    capacity_roads={} # Denominator in the travel time function
    limit_roads={} # Used as maximum allowed cars in a road for the shared constraints
    Road_graph = nx.DiGraph(nx.empty_graph(n_juncs))
    capacity=5
    sigma_0 = capacity/5
    v_free_flow={}
    for i in range(n_juncs): # add self loops
        Road_graph.add_edge(i,i)
    for row in csv_reader:
        # Road_graph.add_edge(int(row[0])-1,int(row[1])-1) # Converting from 1-indexing to 0-indexing
        Road_graph.add_edge(int(row[0]),int(row[1]))
        v_free_flow.update({ ( int(row[0]),int(row[1]) ): row[2] })
    max_nabla_t = 0
    for edge in Road_graph.edges:
        if edge[0]==edge[1]:
            travel_time_derivative_roads.update({edge: 0})
            capacity_roads.update({edge: np.infty})
            limit_roads.update({edge: np.infty})
        else:
            nabla_travel_time = computeTravelTime(capacity, sigma_0, v_free_flow[edge])
            max_nabla_t = max(max_nabla_t, nabla_travel_time)
            travel_time_derivative_roads.update({edge: nabla_travel_time})
            capacity_roads.update({edge: capacity})
            limit_roads.update({edge: capacity})
    nx.set_edge_attributes(Road_graph, values = travel_time_derivative_roads, name = 'nabla_travel_time')
    nx.set_edge_attributes(Road_graph, values = capacity_roads, name = 'capacity')
    nx.set_edge_attributes(Road_graph, values = limit_roads, name = 'limit_roads')

    np.random.seed(1)
    N_edges = len(Road_graph.edges)
    N_city_edges=int(N_edges/5)
    index_city_edges = np.random.randint(0, high=N_edges-1, size=(N_city_edges))
    is_city_road={}
    index=0
    for edge in Road_graph.edges:
        if index in index_city_edges:
            is_city_road.update({edge: True})
        else:
            is_city_road.update({edge: False})
        index = index+1
    nx.set_edge_attributes(Road_graph, values = is_city_road, name = 'is_city_road')
    f= open('manhattan_road_graph.pkl', 'wb')  
    pickle.dump([Road_graph, max_nabla_t], f)
    f.close
