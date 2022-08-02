from collections import deque
import heapq
import numpy as np
import json
import ast

def extract_valid_coords(floorplan):
    '''
    This function receives the floorplan 
    and returns the valid set of coordinates
    agents can initialize at.
    '''
    valid_coord_list = set()
    for i in range(floorplan):
        for j in range(floorplan[0]):
            if floorplan[i][j] == 0:
                # The form (i,j) is compatible with the employee's coords.
                valid_coord_list.add((i,j))
    return valid_coord_list

def load_scanned_floorplan():
    '''
    This function loads the json scan and returns a floorplan
    Input - {"map": {"height": 26, "width": 49, "points": [{"x": 0, "y": 0, "value": 255}, {"x": 1, "y": 0, "value": 255}, {"x": 2, "y": 0, "value": 255}...
    Output- [[1. 1. 1. ... 1. 1. 1.]
            [1. 0. 0. ... 0. 0. 1.]
            [1. 0. 0. ... 0. 0. 1.]
            ...
            [1. 0. 0. ... 0. 0. 1.]
            [1. 0. 0. ... 0. 0. 1.]
            [1. 1. 1. ... 1. 1. 1.]]
    '''
    with open('./data/map.json') as json_file:
        vitaly_floorplan = json.load(json_file)

    floorplan = np.zeros((vitaly_floorplan['map']['width']+1, vitaly_floorplan['map']['height']+1))

    # Defining which coordinates are inaccesible by the agent
    for coor in vitaly_floorplan['map']['points']:
        if coor['value'] == 255:
            floorplan[coor['x']][coor['y']] = 1

    # Accounting for corner cases since we can't change all coordinates on the GUI
    for i in range(len(floorplan)):
        for j in range(len(floorplan[0])):
            if i == len(floorplan)-1:
                floorplan[i][j] = 1
            if j == 26:
                floorplan[i][j] = 1
    return floorplan

def color_walls(numpy_array, allowed_grid):
    '''
    Given the numpy array and the allowed grid, colors the coordinates accordingly to be displayed in RGB form
    Accessible coordinates are black
    Non-accessible coordinates are gray
    '''
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if (i,j) in allowed_grid:
                numpy_array[i][j] = (0,0,0)
            else:
                numpy_array[i][j] = (200,200,200)
    
    return numpy_array

def allowed_floorplan(floorplan):
    '''
    Return a set of allowed points to instantiate from, given a numpy floorplan
    '''
    allowed_set = set()
    for i in range(len(floorplan)):
        for j in range(len(floorplan[0])):
            if floorplan[i][j] == 0:
                allowed_set.add((i,j))
    return allowed_set


def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    '''
    A-star calculation of pairs of (x,y) tuples, given a numpy floorplan.
    Result is in Manhattan distance.
    '''
    array = np.array(array)
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
 
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False

def return_task_loc(task_loc):
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

def tuple_from_string(string):
    return tuple(ast.literal_eval(string))