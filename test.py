from email.errors import ObsoleteHeaderDefect
import numpy as np
import yaml
import json
import sys
import os
import pathlib
import wandb

import matplotlib.pyplot as plt
from customoaas import *
from libr_floorplan import *
from libr_plot import *
from networks import PolicyGradientAgent

# -----------------------------
# - INITIALIZE HYPERPARAMETERS-
# -----------------------------
N_GAMES = 10000
GAMMA = 0.99
LR = 0.0005
#---------------------------
#conf_dict = {"POLG_LR":LR,
#             "GAMMA":GAMMA,
#             "N_GAMES":N_GAMES,
#}
#wandb.init(project="Orchestration AI",config=conf_dict)

# Load data for enviornment
floorplan = load_scanned_floorplan()
db_tasks = load_db_tasks()
db_employees = load_db_employees()
# Preprocess the data
db_tasks = process_db_tasks(db_tasks)
db_employees = process_db_tasks(db_employees)

env = OaasEnv(floorplan, db_tasks, db_employees)

agent = PolicyGradientAgent(gamma=GAMMA, lr=LR, input_dims=[4], n_actions=5)

#fname = 'REINFORCE_' + 'oaas_env' + str(agent.lr) + '_' \
#    + str(N_GAMES) + 'games'
#figure_file = 'plots/'+fname+'.png'

#scores = []

for i in range(N_GAMES):
    done = False
    observation = env.reset()
    score = 0
    iter = 0
    while not done and iter < 100:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action, iter)
        score += reward
        agent.store_rewards(reward)
        observation = observation_
        iter +=1
        env.render(iter)
    #agent.learn()
    #scores.append(score)
    #avg_score = np.mean(scores[-100:])
    #wandb.log({"Scores":score}, commit=True)
    #if i%100 == 0:
    #    print('episode ', i, 'score %.2f' % score, 'average_score %.2f' % avg_score)

#x = [i+1 for i in range(len(scores))]
#plot_learning_curve(scores, x, figure_file)

sys.exit(0)

for employee in env.agent_dict:
    dist = astar(floorplan, env.agent_dict[employee].coords, env.task_dict['task1'].coords)
    print("Length of agent", employee, "=" , len(dist))

env.agent_dict['player1'].move(3)
env.render(1)
