import random
import numpy as np
import sys

class Employee:
    '''
    This class is the employee class.
    Employees Specs:
    - Environment (coordinates to walk through)
    - Task list
    - Assigned task
    '''
    def __init__(self, grid_size, allowed_floorplan):
        self.grid_size = grid_size
        self.allowed_floorplan = allowed_floorplan
        self.reward = 0
        self.coords = random.sample(self.allowed_floorplan,k=1)[0]
        self.x = self.coords[0]
        self.y = self.coords[1]
    
    def move(self,direction):
        if direction == 0: # no movement
            self.x = self.x
            self.y = self.y

        elif direction == 1: # move up
            if self.x <=0:
                self.x = self.x
            else:
                self.x -=1

        elif direction == 2: # move down
            if self.x >=self.grid_size-1:
                self.x = self.x
            else:
                self.x +=1

        elif direction == 3: # move right
            if self.y >=self.grid_size-1:
                self.y = self.y
            else:
                self.y +=1

        elif direction == 4: # move left
            if self.y <= 0:
                self.y = self.y
            else:
                self.y -=1

class Task:
    def __init__(self, grid_size, allowed_floorplan):
        self.allowed_floorplan = allowed_floorplan
        self.reward = 0
        self.coords = random.sample(self.allowed_floorplan,k=1)[0]
        self.x = self.coords[0]
        self.y = self.coords[1]