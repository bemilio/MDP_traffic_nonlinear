from functools import total_ordering
from re import X
import matplotlib as mpl
from cmath import inf

mpl.interactive(True)
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
import networkx as nx
import numpy as np
import pickle
from scipy import sparse


f = open('saved_test_result.pkl', 'rb')
x, dual, residual, Road_graph, edge_time_to_index, node_time_to_index, T_horiz = pickle.load(f)
f.close()

N_iter = residual.size(1)

fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained') 
ax.loglog(np.arange(0, N_iter,20), residual_hsdm[:-1], label="HSDM+PPP")
ax.loglog(np.arange(0, N_iter,20),residual_not_hsdm[:-1], label="PPP")
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.ylim((10**(-6), 10**(-1)))
# plt.title('Residual')
plt.legend()
plt.savefig('Residual.pdf')  


# draw graph
fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained') 
pos=nx.get_node_attributes(Road_graph,'pos')
colors=[]
edgelist=[]
nodes=[]
for edge in Road_graph.edges:
    if not edge[0]==edge[1]:
        color_edge = np.matrix([0])
        for t in range(T_horiz):
            color_edge = np.maximum( S_M[(edge, t)] * sigma_computed, color_edge )
        colors.append(int(206* color_edge)+50)
        edgelist.append(edge)
for node in Road_graph.nodes:
    nodes.append(node)
plt.title("Edges congestion HSDM+PPP")
pos = nx.kamada_kawai_layout(Road_graph)
nx.draw_networkx_nodes(Road_graph, pos=pos, nodelist=nodes, node_size=150, node_color='blue')
nx.draw_networkx_labels(Road_graph, pos)
nx.draw_networkx_edges(Road_graph, pos=pos, edge_color=colors, edgelist = edgelist, edge_cmap=plt.cm.Reds, connectionstyle='arc3, rad = 0.1')

# bar plot of maximum edge congestion compared to constraint
fig, ax = plt.subplots(figsize=(5*1.5, 1.8*1.5), layout='constrained') 
congestions_hsdm ={}
for edge in Road_graph.edges:
    if not edge[0]==edge[1]:
        road_cong = 0
        for t in range(T_horiz):
            road_cong = np.maximum( S_M[(edge, t)] * sigma_computed, road_cong )
        road_cong = road_cong/Road_graph.edges[(edge[0],edge[1])]['limit_roads']
        congestions_hsdm.update({edge: float(road_cong)})
congestions_not_hsdm ={}
for edge in Road_graph.edges:
    if not edge[0]==edge[1]:
        road_cong = 0
        for t in range(T_horiz):
            road_cong = np.maximum( S_M[(edge, t)] * sigma_computed_not_hsdm, road_cong )
        road_cong = road_cong/Road_graph.edges[(edge[0],edge[1])]['limit_roads']
        congestions_not_hsdm.update({edge: float(road_cong)})

plt.bar([k-0.2 for k in range(len(congestions_hsdm))], list(congestions_hsdm.values()), width=0.4, align='center', label="HSDM+PPP")
plt.bar([k+0.2 for k in range(len(congestions_not_hsdm))],  list(congestions_not_hsdm.values()),width=0.4, align='center', label="PPP")
plt.axhline(y=1,linewidth=1, color='red')
plt.xticks(range(len(congestions_not_hsdm)), list(congestions_not_hsdm.keys()), fontsize=6)
plt.legend(prop={'size': 8})
ax.set_xlabel(r'Edge')
ax.set_ylabel(r'$\max_t \sum_{i}  M_{(a,b), i}^t / \bar{c}_{(a,b)}$')
# plt.title('Edges maximum congestion')
plt.savefig('congestion.pdf')  


fig, ax = plt.subplots(figsize=(5*1.5, 1.8*1.5), layout='constrained') 
congestions_hsdm ={}
for edge in Road_graph.edges:
    if not edge[0]==edge[1]:
        road_cong = 0
        for t in range(T_horiz):
            road_cong = ( S_M[(edge, t)] * sigma_computed ) + road_cong
        road_cong = road_cong/Road_graph.edges[(edge[0],edge[1])]['limit_roads']
        congestions_hsdm.update({edge: float(road_cong)})
congestions_not_hsdm ={}
for edge in Road_graph.edges:
    if not edge[0]==edge[1]:
        road_cong = 0
        for t in range(T_horiz):
            road_cong = ( S_M[(edge, t)] * sigma_computed_not_hsdm ) + road_cong
        road_cong = road_cong/Road_graph.edges[(edge[0],edge[1])]['limit_roads']
        congestions_not_hsdm.update({edge: float(road_cong)})

plt.bar([k-0.2 for k in range(len(congestions_hsdm))], list(congestions_hsdm.values()), width=0.4, align='center', label="HSDM+PPP")
plt.bar([k+0.2 for k in range(len(congestions_not_hsdm))],  list(congestions_not_hsdm.values()),width=0.4, align='center', label="PPP")
plt.xticks(range(len(congestions_not_hsdm)), list(congestions_not_hsdm.keys()), fontsize=6)
plt.legend(prop={'size': 8})
ax.set_xlabel(r'Edge')
ax.set_ylabel(r'$\sum_t \sum_{i}  M_{(a,b), i}^t / \bar{c}_{(a,b)}$')
plt.title("Summed congestion of all vehicles over time")
# plt.title('Edges maximum congestion')

sigma_not_electric_hsdm=[]
sigma_not_electric_not_hsdm=[]
test = 0
for agent_id in range(N_agents):
    if not is_electric[agent_id]:
        sigma_not_electric_hsdm = sigma_not_electric_hsdm + x_hsdm[(agent_id, test)]  if len(sigma_not_electric_hsdm) else x_hsdm[(agent_id, test)] 
        sigma_not_electric_not_hsdm = sigma_not_electric_not_hsdm + x_not_hsdm[(agent_id, test)]   if len(sigma_not_electric_not_hsdm) else x_not_hsdm[(agent_id, test)] 

fig, ax = plt.subplots(figsize=(5*1.5, 1.8*1.5), layout='constrained') 
congestions_not_hsdm ={}
for edge in Road_graph.edges:
    if not edge[0]==edge[1]:
        road_cong = 0
        for t in range(T_horiz):
            road_cong = ( S_M[(edge, t)] * sigma_not_electric_not_hsdm ) + road_cong
        road_cong = road_cong
        congestions_not_hsdm.update({edge: float(road_cong)})
plt.bar([k for k in range(len(congestions_not_hsdm))],  list(congestions_not_hsdm.values()),width=0.4, align='center', label="PPP")
plt.xticks(range(len(congestions_not_hsdm)), list(congestions_not_hsdm.keys()), fontsize=6)
plt.legend(prop={'size': 8})
ax.set_xlabel(r'Edge')
ax.set_ylabel(r'$\sum_{t,i}  M_{(a,b), i}^t $')
plt.title("Summed congestion of non-electric vehicles over time")


# bar plot comparison of non-electric vehicles congestion 
fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained') 
congestions_hsdm ={}
congestions_not_hsdm ={}

for edge in Road_graph.edges:
    if Road_graph[edge[0]][edge[1]]['is_city_road']:
        if not edge[0]==edge[1]:
            sum_hsdm_cong = 0
            sum_not_hsdm_cong= 0
            for t in range(T_horiz):
                sum_hsdm_cong = sum_hsdm_cong +  S_M[(edge, t)] * sigma_not_electric_hsdm/Road_graph.edges[(edge[0],edge[1])]['limit_roads']
                sum_not_hsdm_cong = sum_not_hsdm_cong +  S_M[(edge, t)] * sigma_not_electric_not_hsdm/Road_graph.edges[(edge[0],edge[1])]['limit_roads']
            congestions_hsdm.update({edge: float(sum_hsdm_cong)})
            congestions_not_hsdm.update({edge: float(sum_not_hsdm_cong)})

plt.bar([k-0.2 for k in range(len(congestions_hsdm))], list(congestions_hsdm.values()), width=0.4, align='center', label="HSDM+PPP")
plt.bar([k+0.2 for k in range(len(congestions_not_hsdm))],  list(congestions_not_hsdm.values()),width=0.4, align='center', label="PPP")
plt.xticks(range(len(congestions_hsdm)), list(congestions_hsdm.keys()))
ax.set_xlabel(r'Edge')
ax.set_ylabel(r'$\sum_{t, i}  M_{(a,b), i}^t$')
plt.legend()
# plt.title('Expected total non-electric vehicles in city roads')
plt.savefig('non_electrical_congestion.pdf')  



fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained') 

print('Selection function evaluation: HSDM=', cost_hsdm[0].item(), ';   PPP=', cost_not_hsdm[0].item())
congestion_over_time_hsdm=[]
congestion_over_time_not_hsdm=[]
for edge in Road_graph.edges:
    if Road_graph[edge[0]][edge[1]]['is_city_road']:
        if not edge[0]==edge[1]:
            congestion_over_time_hsdm.append( [ (S_M[(edge, t)]*sigma_not_electric_hsdm).item() for t in range(T_horiz)])
            congestion_over_time_not_hsdm.append( [(S_M[(edge, t)] * sigma_not_electric_not_hsdm).item() for t in range(T_horiz)])
total_congestion_hsdm = np.zeros((T_horiz, 1))
total_congestion_not_hsdm = np.zeros((T_horiz, 1))
for i in range(len(congestion_over_time_hsdm)):
    for t in range(T_horiz):
        total_congestion_hsdm[t] = total_congestion_hsdm[t] + congestion_over_time_hsdm[i][t]
        total_congestion_not_hsdm[t]=total_congestion_not_hsdm[t] + congestion_over_time_not_hsdm[i][t]
eps = 10**(-5)
relative_advantage=np.zeros((T_horiz, 1))
plt.plot(total_congestion_not_hsdm - total_congestion_hsdm, label="HSDM" )
# plt.title('City roads congestion of non-electric vehicles')
plt.xlabel('Timestep')
plt.ylabel('\# of cars on city roads')
plt.legend()

cong_hsdm=0
cong_not_hsdm=0
for edge in Road_graph.edges:
    if Road_graph[edge[0]][edge[1]]['is_city_road']:
        for t in range(T_horiz):
            cong_hsdm= cong_hsdm+ ((S_M[(edge, t)]*(sigma_not_electric_hsdm)).item())
            cong_not_hsdm=cong_not_hsdm+ ((S_M[(edge, t)]*sigma_not_electric_not_hsdm).item())



plt.show(block=True)

