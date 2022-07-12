from email.errors import ObsoleteHeaderDefect
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

conf_dict = {"ALGORITHM":'ACTOR_CRITIC',
             "LR":LR,
             "GAMMA":GAMMA,
             "N_GAMES":N_GAMES,
             "FC1_DIMS":FC1_DIMS,
             "FC2_DIMS":FC2_DIMS,
             "EPISODE_LENGTH":EPISODE_LENGTH,
}
# -----------------------------
# --- HYPERPARAMETER TUNING ---
# -----------------------------
sweep_config = {
    'method':'random'
}
metric = {
    'name':'total_reward',
    'goal':'maximize'
}
sweep_config['metric'] = metric
parameters_dict = {
    'fc_layer_size': {
        'values': [16,128,1024]
    },
    'gamma_size': {
        'values': [0.9,0.95,0.99]
    },
    'learning_rate_size': {
        'values': [0.0005, 0.00005, 0.000005]
    },
}
sweep_config['parameters'] = parameters_dict

#pprint.pprint(sweep_config)
#sys.exit(0)

#sweep_id = wandb.sweep(sweep_config, project="Orchestration AI - Sweeps")
#sys.exit(0)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load data for enviornment
floorplan = load_scanned_floorplan()
db_tasks = load_db_tasks()
db_employees = load_db_employees()

db_tasks_long = load_db_tasks_long()
db_employees_long = load_db_employees_long()

print(db_tasks_long)

# Preprocess the data
#env = OaasEnv(floorplan, db_employees, db_tasks)
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

