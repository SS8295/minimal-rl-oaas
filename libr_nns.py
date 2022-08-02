import random
from typing import OrderedDict
import numpy as np
import sys
import math
import networkx as nx
import matplotlib.pyplot as plt
import pprint

def create_initial_mask(task_list_database):
    #print(task_list_database)
    #sys.exit(0)
    mask_dict = {}
    for task in task_list_database:
        mask_dict[task] = 1
    return mask_dict


def max_flow_scheduler(state, sched_int):

    rendering_flag = False

    free_employee_df = state[0].loc[state[0]['agent_state'] == 'READY']
    free_task_df = state[1].loc[state[1]['Status'] == 'READY']

    employees = free_employee_df['AgentID'].tolist()
    tasks = free_task_df['UUID'].tolist()
    priorities = free_task_df['TASK_PRIORITY'].tolist()

    G = nx.Graph(name="match_graph")
    G.add_nodes_from(employees, bipartite=0)
    G.add_nodes_from(tasks, bipartite=1)

    for employee in employees:
        for priority_idx in range(len(priorities)):
            G.add_edge(employee, tasks[priority_idx], weight=priorities[priority_idx])
    
    top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    # bottom_nodes = set(G) - top_nodes

    weights = nx.get_edge_attributes(G,'weight')
    weight_values = nx.get_edge_attributes(G,'weight').values()
    weight_values = [value * 0.5 for value in weight_values]

    pos=nx.bipartite_layout(G,top_nodes)
    matching = nx.max_weight_matching(G, weight='weight')
    output_matching = []
    for match in matching: 
        append_dict = {}
        if match[0] not in top_nodes:
            append_dict['AgentID'] = match[1]
            append_dict['UUID'] = match[0]
        else:
            append_dict['AgentID'] = match[0]
            append_dict['UUID'] = match[1]
        output_matching.append(append_dict)

    if rendering_flag == False:
        return output_matching

    elif rendering_flag == True:
        edge_color_list = ['gray']*len(G.edges)

        for matches in matching:
            for i, edge in enumerate(G.edges()):
                #print(edge, matches)
                #sys.exit(0)
                if edge == matches or (edge[1],edge[0]) == matches:
                    edge_color_list[i] = 'red'

        plt.figure(figsize=(9,9))
        nx.draw(G, 
                node_size=200, 
                #node_color=colors,
                pos=nx.bipartite_layout(G,top_nodes),
                width=list(weight_values),
                with_labels=True,
                edge_color = edge_color_list) #node_color=['red', 'green', 'blue', 'red'], width=list(weight_values),
        #nx.set_node_attributes(G, {**employee_dict, **task_priorities})
        nx.draw_networkx_edge_labels(G,pos, edge_labels=weights)
        x_values, y_values = zip(*pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.5
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.savefig("./graphs/graph_"+str(sched_int)+".png", format="PNG",dpi=200)
        plt.close()
        return output_matching