####################################
# Author: Emil Polakiewicz
# Date: December 2021
# Purpose: Calculate basic statistics about dataset
####################################

#####################
# Imports
#####################

import numpy as np
import pandas as pd
import os

#####################
# Globals
#####################

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '/test_data')


#####################
# Data functions
#####################

# loads entire file
def loadfile(filename):
    if not os.path.exists(os.path.join(DATA_DIR, filename)):
        return np.zeros((0, 10))

    try:
        data = pd.read_csv(os.path.join(DATA_DIR, filename), delimiter=",")
    except pd.errors.EmptyDataError:
        data = np.zeros((0, 10))
    return data


def load_local_file(filename):
    try:
        data = pd.read_csv(os.path.join(BASE_DIR, filename), delimiter=",")
    except pd.errors.EmptyDataError:
        data = np.zeros((0, 0))
    return data

# Loading Files
processes_per_user = load_local_file('processes_per_user.csv')
processes_per_session = load_local_file('processes_per_session.csv')
sessions_per_user = load_local_file('sessions_per_user.csv')


# Calculating statistics
total_proc = sum(processes_per_user[['count']].values)
ave_proc_per_user = total_proc / 10620

total_sessions = sum(sessions_per_user[['num_sessions']].values)
ave_sessions_per_user = total_sessions / 10620

ave_proc_per_session = sum(processes_per_session[['total_processes']].values) / total_sessions

# Printing Statistics
print("Total Processes: " + str(total_proc))
print("Ave Processes Per User: " + str(ave_proc_per_user))

print("Total Sessions: " + str(total_sessions))
print("Ave Sessions Per User: " + str(ave_sessions_per_user))

print("Ave proc per session: " + str(ave_proc_per_session))