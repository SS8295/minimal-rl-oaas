from email.errors import ObsoleteHeaderDefect
from operator import contains
import sched
from telnetlib import GA
import numpy as np
import yaml
import json
import sys
import os
import pathlib
import wandb
import pprint
import torch
import random
from datetime import datetime
import dataframe_image as dfi

import matplotlib.pyplot as plt
from customoaas import *
from libr_floorplan import *
from libr_plot import *
from networks import PolicyGradientAgent
from networks import Agent
from networks import *
from networks_book import *

import torch.optim as optim



# -----------------------------
# - INITIALIZE HYPERPARAMETERS-
# -----------------------------
N_GAMES = 100000
GAMMA = 0.98
LR = 0.00005
FC1_DIMS = 1024
FC2_DIMS = 1
EPISODE_LENGTH = 100

conf_dict = {"ALGORITHM":'WEIGHTED_MAX_FLOW',
             "EPISODE_LENGTH":EPISODE_LENGTH,
}

wand_flag = False

if wand_flag == True:
    wandb.init(project="Orchestration AI - Max Flow Scheduler",config=conf_dict)

# conf_dict = {"ALGORITHM":'ACTOR_CRITIC',
#              "LR":LR,
#              "GAMMA":GAMMA,
#              "N_GAMES":N_GAMES,
#              "FC1_DIMS":FC1_DIMS,
#              "FC2_DIMS":FC2_DIMS,
#              "EPISODE_LENGTH":EPISODE_LENGTH,
# }
# -----------------------------
# --- HYPERPARAMETER TUNING ---
# -----------------------------
# sweep_config = {
#     'method':'random'
# }
# metric = {
#     'name':'total_reward',
#     'goal':'maximize'
# }
# sweep_config['metric'] = metric
# parameters_dict = {
#     'fc_layer_size': {
#         'values': [16,128,1024]
#     },
#     'gamma_size': {
#         'values': [0.9,0.95,0.99]
#     },
#     'learning_rate_size': {
#         'values': [0.0005, 0.00005, 0.000005]
#     },
# }
# sweep_config['parameters'] = parameters_dict

#sweep_id = wandb.sweep(sweep_config, project="Orchestration AI - Sweeps")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dfs = True

# Load data for enviornment
floorplan = load_scanned_floorplan()
db_tasks_long = load_db_tasks_long()
db_employees_long = load_db_employees_long()
curr_time = load_curr_time(db_tasks_long)

TOTAL_SCHED_INTERVALS = len(db_tasks_long)
DAYS = int(TOTAL_SCHED_INTERVALS/96)

#employee_list, task_list = fetch_all_daily_employees_tasks(db_employees_long, db_tasks_long)

env = OaasEnv(floorplan, db_employees_long, db_tasks_long)

#n_agents = env.num_employees
#in_dim = env.num_tasks*6 + env.num_employees*6 + 2
#out_dim = env.num_tasks

#pi = Pi(in_dim, out_dim, fc1_dims = FC1_DIMS)
#optimizer = optim.Adam(pi.parameters(), lr = LR)

task_db_0 = None
task_db_1 = None

for day in range(DAYS):
    for sched_int in range(TOTAL_SCHED_INTERVALS):
        
        curr_time = curr_time
        curr_employees = db_employees_long[sched_int]
        curr_tasks = db_tasks_long[sched_int]

        # Bassam give me current employees and tasks here
        # curr_employees = [] -> live
        # curr_tasks = [] -> live
        
        curr_tasks = pd.DataFrame(curr_tasks)
        curr_employees = pd.DataFrame(curr_employees)

        print("STARTING NEW INTERVAL: ", sched_int, "CURRENT TIME IS ", curr_time)
        print("FETCHING", len(db_employees_long[sched_int]), "EMPLOYEES FROM THE TASK MANAGER")
        print("FETCHING", len(db_tasks_long[sched_int]), "TASKS FROM THE TASK MANAGER")
        
        env.progress_time(curr_time, curr_employees) # Time passage caused this transition

        # Bassam, here is when the newly finished tasks and employees are updates within environment
        # env.finished_tasks has a list of dictionaries that has newly finished tasks, time, and employees
        # 
        # [{'UUID':assignment['UUID'], 'finished_time':curr_time, 'employee':assignment['AgentID']},
        #  {'UUID':assignment['UUID'], 'finished_time':curr_time, 'employee':assignment['AgentID']},
        #   ...
        # ] 

        env.accept_new_tasks(curr_tasks) # Input from task manager caused this transition
        env.accept_new_employees(curr_employees) # Input from task manager caused this transition
        
        state = [env.employee_df, env.task_df]
        #save_input_screenshot(state, sched_int)

        action = max_flow_scheduler(state, sched_int)

        # Bassam goes ahead and updates the databases with the current assignments

        #action = [
        #    {'AgentID': '15320',
        #     'UUID': '64b62b3d-8731-412f-ac40-7712347de8b6'}
        #]

        # short task = e0c5cd98-9329-4017-97b4-031ff35e731a
        # long task = 64b62b3d-8731-412f-ac40-7712347de8b6

        state, reward, done, info = env.step(action, sched_int, curr_time, env.task_df.loc[env.task_df['Status'] == 'READY'])
        
        if wand_flag == True:
            wandb.log({"Collective Distance Traveled":reward},commit=True)
        print('Next state: ----------------------------------------------- ')
        #save_output_screenshot(state, sched_int)

        curr_time = curr_time + timedelta(minutes=15)



        
        
    #loss = train(pi, optimizer, gamma = GAMMA)
    #total_reward = sum(pi.rewards)
    #pi.onpolicy_reset()
    #avg_score = np.mean(loss[-100:].numpy())
    #scores.append(total_reward)

    #if wandb_toggle == True:
    #if epi%1000 == 0:
    #    print(f'episode {epi} loss: {loss}, total_reward: {total_reward}')

    #wandb.log({'total_reward':total_reward}, commit=True)

    #pass

sys.exit(0)
#for i in range(5):

    #db_tasks_long[i]
    #db_employees_long[i]

#for time in range(len(db_employees_long)):
#    print(len(db_employees_long[time]))

# Preprocess the data
env = OaasEnv(floorplan, db_employees_long[i], db_tasks_long[i], i)
#print(summon_graph(db_employees, db_tasks))

#env.matching_step(env.graph)
#env.render(0)
#env.save_video()

sys.exit(0)


#print(env.allowed_floorplan)
#sys.exit(0)

#agent = PolicyGradientAgent(gamma=GAMMA, lr=LR, input_dims=[4], n_actions=5)
#agent = Agent(gamma=0.99, lr=LR, input_dims=[4], n_actions = 5, fc1_dims = FC1_DIMS, fc2_dims=FC2_DIMS)

#fname = 'ACTOR_CRITIC_' + 'oaas_env_' + str(agent.fc1_dims) + '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_' \
#        + str(agent.lr) + '_lr_' + str(N_GAMES) + 'games'

#figure_file = 'plots/'+fname+'.png'

scores = []

in_dim = int(env.observation_space.shape[0])
out_dim = int(env.action_space.n)
pi = Pi(in_dim, out_dim, fc1_dims = FC1_DIMS)
optimizer = optim.Adam(pi.parameters(), lr = LR)

for epi in range(N_GAMES):
    state = env.reset()
    for t in range(EPISODE_LENGTH):
        action = pi.act(state)
        state, reward, done, _ = env.step(action, t)
        pi.rewards.append(reward)
        if done:
            break
    loss = train(pi, optimizer, gamma = GAMMA)
    total_reward = sum(pi.rewards)
    pi.onpolicy_reset()
    #avg_score = np.mean(loss[-100:].numpy())
    scores.append(total_reward)

    #if wandb_toggle == True:
    if epi%1000 == 0:
        print(f'episode {epi} loss: {loss}, total_reward: {total_reward}')

    wandb.log({'total_reward':total_reward}, commit=True)
#x = [i+1 for i in range(len(scores))]
#plot_learning_curve(scores, x, figure_file)

