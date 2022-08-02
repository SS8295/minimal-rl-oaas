# Gym-related imports
from math import floor
from venv import create

from gym import Env
from gym import logger, spaces
#from gym.spaces import Discrete, Box

# Minimal standard imports
import sys
import time 
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import glob
from datetime import datetime
import dataframe_image as dfi

# Local imports
from libr import *
from libr_data import *
from libr_floorplan import *
from libr_nns import *
from libr_plot import *

class OaasEnv(Env):

    def __init__(self, floorplan, all_time_employees, all_time_tasks):
        
        # Database Parameters
        self.employee_list, self.task_list = fetch_all_daily_employees_tasks(all_time_employees, all_time_tasks)
        self.employee_df = create_employee_df(self.employee_list)
        self.task_df = create_task_df(self.task_list)

        # Grid Configuration
        self.floorplan = floorplan
        self.grid_size = len(floorplan)
        self.allowed_floorplan = allowed_floorplan(floorplan)      
 
        # Numerical Initializations
        self.num_employees = len(self.employee_list)
        self.num_tasks = len(self.task_list)
        
        # Variables saved for data analysis
        self.curr_assignments = []
        self.finished_tasks = []

    def progress_time(self, curr_time, curr_employees):
        '''
        Takes the current (passed) time and the head count of the store and updates the status
        of the tasks based on the confluence diagram.

        Inputs:     Current time                    datetime()
                    Current Employees               pd.Dataframe()
        '''
        
        for assignment in self.curr_assignments:

            employee_present = False
            finish_threshold = False

            # Preproceess employee_present and finish_threshold flags
            print("ASSIGNMENT = ",assignment)
            #print(assignment['TIME_ASSIGNED'])
            #print("assignment duration = ",assignment['TASK_DURATION'].item())
            #print(assignment['TASK_DURATION'].item())
            #print(timedelta(minutes = assignment['TASK_DURATION'].item()))
            #print(assignment['TIME_ASSIGNED'] + timedelta(minutes = assignment['TASK_DURATION'].item()))
            #print(curr_time)
            if assignment['TIME_ASSIGNED'] + timedelta(minutes = assignment['TASK_DURATION'].item()) <= curr_time:
                finish_threshold = True

            if assignment['AgentID'] in curr_employees['AgentID'].values:
                employee_present = True
            
            # Change status of task if employee left without finishing
            # TASK:         PROCESSING -> READY
            # TASK:         PROCESSING -> MISSED
            # EMPLOYEE:     BUSY -> -
            if not employee_present and not finish_threshold:
                print('Employee ', assignment['AgentID'], 'left without finihsing the task!')
                self.curr_assignments.remove(assignment)
                self.task_df.loc[self.task_df['UUID']==assignment.get('UUID'),'Status'] = 'READY'

            # If time elapsed, mask task completed. For now, it does not matter if the employee left or not (t < 15min)
            elif finish_threshold:
                print('Task ', assignment['UUID'], ' finished!')
                self.finished_tasks.append({'UUID':assignment['UUID'], 'finished_time':curr_time, 'employee':assignment['AgentID']})
                self.curr_assignments.remove(assignment)
                self.task_df.loc[self.task_df['UUID']==assignment.get('UUID'),'Status'] = 'FINISHED'
                self.employee_df.loc[self.employee_df['AgentID']==assignment.get('AgentID'),'agent_state'] = 'READY'

            # Task still in progress, no change in the status of the employee or task df entry 
            else:
                continue

    def accept_new_tasks(self, curr_tasks):
        '''
        This function populates the self.task_df with tasks and changes their status to READY
        while also incorporating the metadata from the task manager.

        Input:     Tasks                        pd.Dataframe()
        '''

        for curr_task in range(len(curr_tasks)):
            
            if self.task_df.loc[self.task_df['UUID']==curr_tasks.loc[curr_task].get('UUID')]['Status'].item() == 'OUT':
                self.task_df.loc[self.task_df['UUID']==curr_tasks.loc[curr_task].get('UUID'),'Status'] = 'READY'
                for metadata in curr_tasks:
                    if metadata == 'TASK_LOCATION':
                        self.task_df.loc[self.task_df['UUID']==curr_tasks.loc[curr_task].get('UUID'),metadata] = return_task_loc(curr_tasks.loc[curr_task].get(metadata))
                    else:
                        self.task_df.loc[self.task_df['UUID']==curr_tasks.loc[curr_task].get('UUID'),metadata] = curr_tasks.loc[curr_task].get(metadata)

            elif self.task_df.loc[self.task_df['UUID']==curr_tasks.loc[curr_task].get('UUID')]['Status'].item() == 'READY':
                for metadata in curr_tasks:
                    if metadata == 'TASK_LOCATION':
                        self.task_df.loc[self.task_df['UUID']==curr_tasks.loc[curr_task].get('UUID'),metadata] = return_task_loc(curr_tasks.loc[curr_task].get(metadata))
                    else:
                        self.task_df.loc[self.task_df['UUID']==curr_tasks.loc[curr_task].get('UUID'),metadata] = curr_tasks.loc[curr_task].get(metadata)

    def accept_new_employees(self, curr_employees):
        '''
        This function populates the self.employee_df with employees and changes status of fresh 
        new employees that haven't been assigned a task to READY and to employees that left the store 
        to OUT while also incorporating the metadata from the task manager.

        Input:     Employees                    pd.Dataframe()
        '''
        #self.employee_df = create_employee_df(self.employee_list)
        self.employee_df['Location'] = self.employee_df['Location'].astype('object')

        for curr_employee in range(len(curr_employees)):
            
            curr_employee_id = curr_employees.loc[curr_employee].get('AgentID')
            
            if self.employee_df.loc[self.employee_df['AgentID']==curr_employee_id]['agent_state'].item() == 'OUT':
                self.employee_df.loc[self.employee_df['AgentID']==curr_employee_id,'agent_state'] = 'READY'
                self.employee_df.loc[self.employee_df['AgentID']==curr_employee_id,'Location'] = str(curr_employees.loc[curr_employee,'Location'])
                self.employee_df.loc[self.employee_df['AgentID']==curr_employee_id,'Skillset'] = 'ALL'

    def step(self, action, sched_int, curr_time, curr_tasks):
        '''
        Takes a step in the environment.

        Inputs:     Employee/Task Matching      [dict(), dict(), ... , dict()]
                    Schedule interval           int()
        Outputs:    State, reward, done, info   [pd.DataFrame(), pd.DataFrame()], float(), bool(), dict()
        '''

        reward = 0

        for assignment in action:
            new_task_flag = True

            for prev_assignments in self.curr_assignments:
                if assignment['UUID'] == prev_assignments['UUID']:
                    new_task_flag = False
        
            if new_task_flag == False:
                print("[WRONG ACTION]: Attempting to do a task that is already being done!")
                #reward=-10

            elif self.task_df.loc[self.task_df['UUID']==assignment['UUID']]['Status'].item() == 'FINISHED':
                print("[WRONG ACTION]: Attempting to do a finished task!")
                #reward=-10

            else:
                self.curr_assignments.append({'UUID': assignment['UUID'], 'AgentID': assignment['AgentID'], 'TIME_ASSIGNED':curr_time,
                                                'TASK_DURATION':curr_tasks.loc[curr_tasks['UUID']==assignment['UUID'],'TASK_DURATION'].values[0]})
                                                
                self.task_df.loc[self.task_df['UUID']==assignment.get('UUID'),'Status'] = 'PROCESSING'
                self.employee_df.loc[self.employee_df['AgentID']==assignment.get('AgentID'),'agent_state'] = 'BUSY'
                
                empl_coord = self.employee_df.loc[self.employee_df['AgentID']==assignment.get('AgentID'),'Location'].item()
                task_coord = self.task_df.loc[self.task_df['UUID']==assignment.get('UUID'),'TASK_LOCATION'].item()
                #print(task_coord)
                #sys.exit(0)
                empl_tuple_coord = tuple_from_string(empl_coord)
                print("trying with: ",task_coord)
                task_tuple_coord = tuple_from_string(task_coord)

                dist_penalty = len(astar(self.floorplan, empl_tuple_coord, task_tuple_coord))
                print("Employee walked from ", empl_tuple_coord, "to ",task_tuple_coord, "covering a distance of ",dist_penalty)
                self.employee_df.loc[self.employee_df['AgentID']==assignment.get('AgentID'),'Location'] = self.task_df.loc[self.task_df['UUID']==assignment.get('UUID'),'TASK_LOCATION'].item()
                #print(self.employee_df.loc[self.employee_df['AgentID']==assignment.get('AgentID'),'Location'])
                #sys.exit(0)
                #print(self.task_df.loc[self.task_df['UUID']==assignment.get('UUID')])
                #print(self.employee_df.loc[self.employee_df['AgentID']==assignment.get('AgentID')])
                #print("Assingment history = ",self.curr_assignments)   
                reward-=dist_penalty         
        
        new_env_state = []
        new_env_state.append(self.employee_df)
        new_env_state.append(self.task_df)

        done = False
        info = {}

        return new_env_state, reward, done, info

    def matching_step(self, matching):
        horizon = 30

        for instance in range(horizon):
            print('Step ', instance)
            self.render(instance)
            for match in matching:
                
                if len(match[0]) <= len(match[1]):
                    employee = match[0]
                    task = match[1]
                else:
                    employee = match[1]
                    task = match[0]
                #print('employee: ',employee)
                #print('task: ',task)
                #print(self.employee_dict[employee].coords, self.task_dict[task].coords)
                path = astar(self.floorplan, self.employee_dict[employee].coords, self.task_dict[task].coords)
                #print(path)
                if len(path) > 0:
                    self.employee_dict[employee].coords = path[-1]
                else:
                    pass

    def render(self,iter):
        #print("rendering...")
        fig,ax = plt.subplots(1,2)

        fig.subplots_adjust(bottom=0.4)
        colorstate = np.zeros((self.grid_size, self.grid_size,3),dtype=np.uint8)
        colorstate = color_walls(colorstate, self.allowed_floorplan)

        # Render Employees      
        for employee_int in range(self.num_employees):
            colorstate[self.employee_dict[self.db_employees['AgentID'][employee_int]].coords[0]][self.employee_dict[self.db_employees['AgentID'][employee_int]].coords[1]] = (150, 55, 255)

        # Render Tasks      
        for task_int in range(self.num_tasks):
            colorstate[self.task_dict[self.db_tasks['UUID'][task_int]].coords[0]][self.task_dict[self.db_tasks['UUID'][task_int]].coords[1]] = self.task_dict[self.db_tasks['UUID'][task_int]].color

        # Render Tasks 
        #if self.task_dict      
        #colorstate[self.task_dict['task1'].coords[0]][self.task_dict['task1'].coords[1]] = (255, 165, 0)
        ax.imshow(colorstate, interpolation='None')
        ax.axis('off')

        #for employee_int in range(self.num_employees):
        #    x_coord = self.employee_dict[self.db_employees['AgentID'][employee_int]].coords[0]
        #    y_coord = self.employee_dict[self.db_employees['AgentID'][employee_int]].coords[1]
            #dist = astar(self.floorplan, self.employee_dict['player'+str(player)].coords, self.task_dict['task1'].coords)
            #fig.text(0.3,0.3-0.05*player,f'Employee {player}: ({x_coord},{y_coord})     Distance to task = {len(dist)}')
            #fig.text(0.5,0.25-0.05*player,f'Distance to task: ',len(astar(self.floorplan, self.employee_dict['player'+str(player)].coords, self.task_dict['task1'].coords)))
       
        #(x_task, y_task) = self.task_dict['task1'].coords
        #fig.text(0.3,0.05,f'Task coords: ({x_task},{y_task})')
        #fig.text(0.3,0.10,f'Targets: ({self.target1_x},{self.target1_y}) and ({self.target2_x},{self.target2_y})')
        fig.text(0.55,0.25, f'Iteration: {iter}')
        #plt.show()
        plt.savefig(fname='./renders/live_render'+str(iter),dpi=300)
        fig,ax = render_mpl_table(self.employee_df, header_columns=0, col_width=2.0)
        fig.savefig("table_mpl.png")
        plt.close()

    def reset(self):    

        self.time = 0
        #self.curr_assignments = {}
        return self.obs()

    def obs(self):

        self.observation = []

        self.observation.append(self.employee_dict['player'+str(0)].coords[0])
        self.observation.append(self.employee_dict['player'+str(0)].coords[1])
        self.observation.append(self.task_dict['task1'].coords[0])
        self.observation.append(self.task_dict['task1'].coords[1])

        return self.observation