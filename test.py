from customoaas import *
import numpy as np
import yaml
from libr_floorplan import *

#floorplan = [[0,0,0],[1,0,1],[0,0,0]]

#start = (0,0)
#end = (2,2)


#path = astar(floorplan, start, end)
#print(path)

with open('config.yaml') as f:
    floorplan = yaml.safe_load(f)

env = OaasEnv(floorplan)
env.render(0)

#if (1,5) in allowed_floorplan:
#    print("were in")
#new_task_coords = (4,5)
#print(env.agent_dict['player0'].coords)
#print(floorplan)
#sys.exit(0)

#print(floorplan)
#start = (1,1)
#end = (3,2)
#print(minPushBox(floorplan, start, end))
#sys.exit(0)

#path = astar(np.array(floorplan), (1,1), (1,7))
#path = astar(floorplan, (8,7), (1,7))
#path = astar(floorplan, (6,5), (1,7))
#path = astar(floorplan, (8,8), (1,7))
#path = astar(floorplan, (1,5), (1,7))


for employee in env.agent_dict:
    dist = astar(floorplan, env.agent_dict[employee].coords, env.task_dict['task1'].coords)
    print("Length of agent", employee, "=" , len(dist))
