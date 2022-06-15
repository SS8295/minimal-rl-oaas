import random

class TeamPlayer:
    def __init__(self,grid_size):
        self.grid_size = grid_size
        self.reward = 0
        self.x = random.randint(0, grid_size-1) #grid_size-1
        self.y = random.randint(0, grid_size-1) #grid_size-1
    
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