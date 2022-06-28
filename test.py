from customoaas import *
import numpy as np
import yaml
from libr_floorplan import *
import json
import sys
import os
import pathlib


#print('getcwd:      ', os.getcwd())
#print('__file__:    ', __file__)
#print(os.getcwd())
#sys.exit(0)

# Opening JSON file
with open('map.json') as json_file:
    vitaly_floorplan = json.load(json_file)

#print(vitaly_floorplan['map']['height'])
#print(vitaly_floorplan['map']['width'])

#print(vitaly_floorplan['map']['height'])

floorplan = np.zeros((vitaly_floorplan['map']['width']+1, vitaly_floorplan['map']['height']+1))
#print(floorplan.shape)
#sys.exit(0)

for coor in vitaly_floorplan['map']['points']:

    #print(coor)
    if coor['value'] == 255:
    #    print(coor)
    #    print('x = ',coor['x'])
    #    print('y = ',coor['y'])
        floorplan[coor['x']][coor['y']] = 1

#print(floorplan)
#sys.exit(0)

#print(diogos_floorplan[0])
#sys.exit(0)

env = OaasEnv(floorplan)
env.render(0)

for employee in env.agent_dict:
    dist = astar(floorplan, env.agent_dict[employee].coords, env.task_dict['task1'].coords)
    print("Length of agent", employee, "=" , len(dist))

env.agent_dict['player1'].move(3)
env.render(1)
