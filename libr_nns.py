import random
import numpy as np
import sys
import math

def create_initial_mask(task_list_database):
    #print(task_list_database)
    #sys.exit(0)
    mask_dict = {}
    for task in task_list_database:
        mask_dict[task] = 1
    return mask_dict