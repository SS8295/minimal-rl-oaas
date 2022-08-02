import random
import numpy as np
import sys
import math

class Employee:
    '''
    This class is the employee class.
    Employees Specs:
    - Environment (coordinates to walk through)
    - Task list
    - Assigned task
    '''
    def __init__(self, allowed_floorplan, employee_int, db_employees):
        
        #print(db_employees['Location'][employee_int])
        self.allowed_floorplan = allowed_floorplan
        #self.coords = random.sample(self.allowed_floorplan,k=1)[0]
        self.coords = (db_employees['Location'][employee_int][0], db_employees['Location'][employee_int][1])
    
    def move(self,direction):
        if direction == 0: # no movement
            pass
        elif direction == 1: # move up
            if (self.coords[0]-1, self.coords[1]) not in self.allowed_floorplan:
                self.coords[0] = self.coords[0]
            else:
                self.coords[0] -=1
        elif direction == 2: # move down
            if (self.coords[0]+1, self.coords[1]) not in self.allowed_floorplan:
                self.coords[0] = self.coords[0]
            else:
                self.coords[0] +=1
        elif direction == 3: # move right
            if (self.coords[0], self.coords[1]+1) not in self.allowed_floorplan:
                self.coords[1] = self.coords[1]
            else:
                self.coords[1] +=1
        elif direction == 4: # move left
            if (self.coords[0], self.coords[1]-1) not in self.allowed_floorplan:
                self.coords[1] = self.coords[1]
            else:
                self.coords[1] -=1

class Task:
    def __init__(self, allowed_floorplan, task_int, db_tasks):
        #print(self.return_task_loc(db_tasks['TASK_LOCATION'][task_int]))
        #sys.exit(0)
        self.allowed_floorplan = allowed_floorplan
        #self.reward = 0
        #self.coords = random.sample(self.allowed_floorplan,k=1)[0]
        self.coords = self.return_task_loc(db_tasks['TASK_LOCATION'][task_int])
        self.priority = db_tasks['TASK_PRIORITY'][task_int]
        self.color = self.color_from_priority(self.priority)

    def color_from_priority(self,priority):

        baseline = 50
        rate = 2
        red_channel = 255-(255-baseline)*rate**(-priority)
        green_channel = baseline
        blue_channel = baseline
        return (red_channel, green_channel, blue_channel)

    def return_task_loc(self, task_loc):
        MINIGRID_ZFL_AREA_COORDINATES = {
        'CB' :str([3, 2]),
        'BCF' :str([1, 13]),
        'FS' : str([13, 3]),
        'FESF' :str([17, 5]),
        'OS' :str([14, 12]),
        'BR' :str([7, 6]),
        'CR' :str([13, 9]),
        'RTM' :str([8, 10])
        }
        return MINIGRID_ZFL_AREA_COORDINATES[task_loc]
