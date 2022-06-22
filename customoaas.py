# Gym-related imports
from math import floor
from gym import Env
from gym import logger, spaces
#from gym.spaces import Discrete, Box

# Minimal standard imports
import sys
import time 
import random
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from libr import *
from libr_rl import *
from libr_floorplan import *

class OaasEnv(Env):

    def __init__(self, floorplan):
        
        # Grid Size
        #self.floorplan = floorplan
        self.grid_size = len(floorplan) #len(floorplan)
        self.allowed_floorplan = allowed_floorplan(floorplan)
        #print(self.allowed_floorplan)
        #sys.exit(0)
        #Agents in Environment
        self.num_agents = 5
        self.num_tasks = 1
        # Action Enumeration
        self.num_actions = 5

        # Employee Summoning
        self.agent_dict = dict()
        for player_int in range(self.num_agents):
            self.agent_dict['player'+str(player_int)] = Employee(self.grid_size, self.allowed_floorplan)
            #print(self.agent_dict['player'+str(player_int)].coords)
        #sys.exit(0)
        for key, value in self.agent_dict.items() :
            print(key, value)
        #sys.exit(0)

        # Task Summoning
        self.task_dict = dict()
        self.task_dict['task1'] = Task(self.grid_size, self.allowed_floorplan)

        # Observation and Action Space
        low = np.zeros(2*(self.num_agents))
        high = np.full(2*(self.num_agents), self.grid_size)
        self.action_space = []
        self.observation_space = []
        for _ in range(self.num_agents):
            self.action_space.append(spaces.Discrete(5))
            self.observation_space.append(spaces.Box(np.float32(low), np.float32(high)))
        print("Observation Space = ", self.observation_space)
        print("Action Space = ",self.action_space)
        #self.observation_size = self.obs()
        #self.observation_size = len(self.observation_size)
        #print(self.observation_size)

    def step(self, action_list):

        self.done_flag = False
        
        # Move agents in environment
        for act in range(len(action_list)):
            self.agent_dict['player'+str(act)].move(action_list[act])

        reward_list = [0,0,0]
        #aux_rew = self.aux_reward()*self.aux_multiplier
        #print('Aux Reward = ',-aux_rew)
        #reward_list = [-aux_rew, -aux_rew]

        if self.team_wins() and self.adv_wins():
            reward_list = [0,0,0]
            self.done_flag = True

        if self.team_wins():
            reward_list = [1,1,-1]
            self.done_flag = True
        
        if self.adv_wins():
            reward_list = [-1,-1,1]
            self.done_flag = True
        
        return self.obs(), reward_list, self.done_flag, {}

    def render(self,iter):
        print("I will render")
        fig,ax = plt.subplots(1)
        fig.subplots_adjust(bottom=0.3)
        colorstate = np.zeros((self.grid_size, self.grid_size,3),dtype=np.uint8)
        colorstate = color_walls(colorstate, self.allowed_floorplan)

        # Render Agents      
        for player_int in range(self.num_agents):
            colorstate[self.agent_dict['player'+str(player_int)].x][self.agent_dict['player'+str(player_int)].y] = (150, 55, 255)

        # Render Tasks      
        colorstate[self.task_dict['task1'].x][self.task_dict['task1'].y] = (255, 0, 0)
        #colorstate[self.agent_dict['player'+str(self.num_agents)].x][self.agent_dict['player'+str(self.num_agents)].y] = (255, 50, 50)
        ax.imshow(colorstate, interpolation='None')
        ax.axis('off')

        for player in range(self.num_agents):
            x_coord = self.agent_dict['player'+str(player)].x
            y_coord = self.agent_dict['player'+str(player)].y
            fig.text(0.3,0.25-0.05*player,f'Player {player}: ({x_coord},{y_coord})')
        #x_coord_adv = self.agent_dict['player'+str(self.num_agents)].x
        #y_coord_adv = self.agent_dict['player'+str(self.num_agents)].y
        #fig.text(0.3,0.15,f'Adversary: ({x_coord_adv},{y_coord_adv})')
        #fig.text(0.3,0.10,f'Targets: ({self.target1_x},{self.target1_y}) and ({self.target2_x},{self.target2_y})')
        fig.text(0.55,0.25, f'Iteration: {iter}')
        #plt.savefig(f'saved_images/out_{iter}.png')
        plt.savefig(fname='./renders/live_render')

    def reset(self):

        # Player Summoning
        self.agent_dict = dict()
        for player_int in range(self.num_agents):
            self.agent_dict['player'+str(player_int)] = TeamPlayer(self.grid_size)
        self.agent_dict['player'+str(self.num_agents)] = AdvPlayer(self.grid_size)

        if self.team_wins() or self.adv_wins():
            self.done_flag = True
        else:
            self.done_flag = False

        return self.obs()
    
    def staged_reset(self, coords):

        for player_int in range(self.num_agents):
            self.agent_dict['player'+str(player_int)] = TeamPlayer(self.grid_size)
            self.agent_dict['player'+str(player_int)].x = coords[2*player_int]
            self.agent_dict['player'+str(player_int)].y = coords[2*player_int+1]
        self.agent_dict['player'+str(self.num_agents)] = AdvPlayer(self.grid_size)
        self.agent_dict['player'+str(self.num_agents)].x = coords[2*self.num_agents]
        self.agent_dict['player'+str(self.num_agents)].y = coords[2*self.num_agents+1]

        if self.team_wins() or self.adv_wins():
            self.done_flag = True
        else:
            self.done_flag = False

        return self.obs()

    def obs(self):
        observation_space = []
        for player_int in range(self.num_agents):
            observation_space.append(self.agent_dict['player'+str(player_int)].x)
            observation_space.append(self.agent_dict['player'+str(player_int)].y)
        return observation_space # [x,y]

    def print_coords(self):
        for player in range(self.num_agents):
            x_coord = self.agent_dict['player'+str(player)].x
            y_coord = self.agent_dict['player'+str(player)].y
            print(f'Player {player}: ({x_coord},{y_coord})')
        print('--------------')
        #x_coord_adv = self.agent_dict['player'+str(self.num_agents)].x
        #y_coord_adv = self.agent_dict['player'+str(self.num_agents)].y
        print(f'Adversary: ({x_coord_adv},{y_coord_adv})')
        print(f'Targets: ({self.target1_x},{self.target1_y}) and ({self.target2_x},{self.target2_y})')
        print('--------------')
 

        x=0
        y=1
        coords = [self.agent_dict['player'+str(self.num_agents)].x, self.agent_dict['player'+str(self.num_agents)].y]

        if coords[x] == self.target1_x and coords[y] == self.target1_y:
            return True
        if coords[x] == self.target2_x and coords[y] == self.target2_y:
            return True
        
        return False