import numpy as np
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
import pprint
from networkx.algorithms import bipartite
from itertools import count
from datetime import datetime, timedelta


# Loading Data to the custom gym environment

def load_db_tasks():
    with open('./data/tasks.pickle', 'rb') as handle:
        db_task_list = pickle.load(handle)
    return db_task_list

def load_db_employees():
    with open('./data/agents.pickle', 'rb') as handle:
        db_employees = pickle.load(handle)
    return db_employees

def load_db_employees_long():
    with open('./data/one_month_interval_employees.pkl', 'rb') as handle:
        db_employees_long = pickle.load(handle)
    return db_employees_long

def load_db_tasks_long():
    with open('./data/one_month_interval_tasks.pkl', 'rb') as handle:
        db_tasks_long = pickle.load(handle)

    for interval in range(len(db_tasks_long)):
        for task in db_tasks_long[interval]:
            #print(type(task['TASK_CREATION_TIME']))
            temp_var = str(int(''.join(c for c in task['TASK_CREATION_TIME'] if c.isdigit())))
            yr = int(temp_var[0:4])
            mt = int(temp_var[4:6])
            dy = int(temp_var[6:8])
            hr = int(temp_var[8:10])
            mn = int(temp_var[10:12])
            sc = int(temp_var[12:14])
            date_time = datetime(year=yr, month=mt, day=dy, hour=hr, minute=mn, second=sc)
            task['TASK_CREATION_TIME'] = date_time
            #sys.exit(0)
    return db_tasks_long

def load_curr_time(db_tasks_long):
    return db_tasks_long[0][0]['TASK_CREATION_TIME']

def fetch_all_daily_employees_tasks(db_employees_long, db_tasks_long):

    day_task_set = set()
    day_employee_set = set()
    for time_step in range(len(db_tasks_long)):
        task_step = db_tasks_long[time_step]
        employee_step = db_employees_long[time_step]
        for task in task_step:
            day_task_set.add(task['UUID'])
        #print(employee_step)
        for employee in employee_step:
            day_employee_set.add(employee['AgentID'])

    return list(day_employee_set), sorted(list(day_task_set))

# Processing data in the gym environment
def db_to_pandas(db_tasks):
    return pd.DataFrame(data=db_tasks)

def create_employee_df(employee_list):
    availability_list = []
    location_information = []
    skillset_information = []
    for _ in range(len(employee_list)):
        availability_list.append('OUT')
        location_information.append(None)
        skillset_information.append(None)
    ret_df = pd.DataFrame(list(zip(employee_list, availability_list, location_information, skillset_information)), columns =['AgentID', 'agent_state', 'Location', 'Skillset'])
    #print(ret_df)
    return ret_df

def create_task_df(task_list):

    availability_list = []
    location_information = []
    expected_minutes_left = []
    task_priority = []
    task_duration = []
    task_creation_time = []

    for _ in range(len(task_list)):
        availability_list.append('OUT')
        location_information.append(None)
        expected_minutes_left.append(None)
        task_priority.append(None)
        task_duration.append(None)
        task_creation_time.append(None)

    ret_df = pd.DataFrame(list(zip(task_list, availability_list, location_information, expected_minutes_left, task_priority, task_duration, task_creation_time)), columns =['UUID', 'Status', 'TASK_LOCATION', 'EXPECTED_MINUTE_LEFT', 'TASK_PRIORITY', 'TASK_DURATION', 'TASK_CREATION_TIME'])
    #print(ret_df)
    #sys.exit(0)
    return ret_df

def summon_graph(time_instance, db_employees_load, db_tasks_load):

    #print(db_tasks_load)
    #sys.exit(0)


    employee_dict = {}
    for i in range(len(db_employees_load)):
        #db_employees_load[i]['type'] = 'employee'
        employee_dict[db_employees_load[i]['AgentID']] = db_employees_load[i]
    task_dict = {}
    for i in range(len(db_tasks_load)):
        #db_tasks_load[i]['type'] = 'task'
        task_dict[db_tasks_load[i]['UUID']] = db_tasks_load[i]
    
    G = nx.Graph(name="match_graph")
    G.add_nodes_from(employee_dict, bipartite=0)
    G.add_nodes_from(task_dict, bipartite=1)
    nx.set_node_attributes(G, {**employee_dict, **task_dict})

    edge_list = []
    for employee in employee_dict:
        for task in task_dict:
            G.add_edge(task,employee, weight = task_dict[task]['TASK_PRIORITY'])

    top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes = set(G) - top_nodes

    weights = nx.get_edge_attributes(G,'weight')
    weight_values = nx.get_edge_attributes(G,'weight').values()
    weight_values = [value * 0.5 for value in weight_values]

    
    #colors = nx.get_edge_attributes(G,'weight')

    #groups = set(nx.get_node_attributes(G,'type').values())
    #mapping = dict(zip(sorted(groups),count()))
    #nodes = G.nodes()
    #colors = [mapping[G.node[n]['type']] for n in nodes]

    
    #sys.exit(0)

    pos=nx.bipartite_layout(G,top_nodes)
    #return  G

    #nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, 
    #                        with_labels=False, node_size=100, cmap=plt.cm.jet)
    #, pos=nx.bipartite_layout(G,employee_list))

    matching = nx.max_weight_matching(G, weight='weight')
    print("Proposed matching: ",matching)

    edge_color_list = ['grey']*len(G.edges)
    #replace the color in edge_color_list with red if the edge belongs to the shortest path:
    
    for matches in matching:
        for i, edge in enumerate(G.edges()):
            #print(edge, matches)
            #sys.exit(0)
            if edge == matches or (edge[1],edge[0]) == matches:
                edge_color_list[i] = 'red'

    nx.draw(G, 
            node_size=200, 
            #node_color=colors,
            pos=nx.bipartite_layout(G,top_nodes),
            width=list(weight_values),
            with_labels=True,
            edge_color = edge_color_list) #node_color=['red', 'green', 'blue', 'red'], width=list(weight_values),

    nx.draw_networkx_edge_labels(G,pos, edge_labels=weights)
    
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.5
    plt.figure(figsize=(9,9))
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.savefig("./graphs/graph"+str(time_instance)+".png", format="PNG",dpi=1000)
    print('FINISHEEEED')
    sys.exit(0)
    return matching



def task_coloring_array(db_employees_long, db_tasks_long, task_list):

    ret_array = []
    for interval in range(len(db_tasks_long)):
        task_list_interval = np.zeros(len(task_list))
        for task in range(len(db_tasks_long[interval])):
            #pprint.pprint(task_list)
            #print(db_tasks_long[interval][task]['UUID'])
            task_list_interval[task_list.index(db_tasks_long[interval][task]['UUID'])]=1
        ret_array.append(task_list_interval)
    #ret_array = [[0,0,0,1], [1,0,0,1], [1,0,1,1], [1,1,1,1]]
    #plt.imshow(np.transpose(np.array(ret_array)), interpolation='none', aspect='auto')#, interpolation='nearest')

    #plt.savefig(fname='./tasks')
    #plt.close()
    return ret_array

def plot_2D(input_2D_array, filename):
    plt.imshow(np.transpose(np.array(input_2D_array)), interpolation='none', aspect='auto')
    plt.savefig(fname='./'+filename)
