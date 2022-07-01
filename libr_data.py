import numpy as np
import pickle
import pandas as pd

# Loading Data to the custom gym environment

def load_db_tasks():
    with open('./data/tasks.pickle', 'rb') as handle:
        db_task_list = pickle.load(handle)
    return db_task_list

def load_db_employees():
    with open('./data/agents.pickle', 'rb') as handle:
        db_employees = pickle.load(handle)
    return db_employees

# Processing data in the gym environment
def process_db_tasks(db_tasks):
    return pd.DataFrame(data=db_tasks)
