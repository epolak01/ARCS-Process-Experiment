####################################
# Author: Emil Polakiewicz
# Date: December 2021
# Purpose: Filter process file to make it smaller
####################################

#####################
# Imports
#####################

import pandas as pd
import os

#####################
# Globals
#####################

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../../datasets/kent2016')
chunk_size = 20000000
i = 0

#####################
# Filter Proc File
#####################

# load file into chunks
for proc_chunk in pd.read_csv(os.path.join(DATA_DIR, "proc.txt.gz"), delimiter=",", chunksize=chunk_size,
                              skiprows=0):

    proc_chunk.columns = ['time', 'user', 'comp', 'process', 'start/end']

    # Only keep real user account data
    proc_chunk = proc_chunk[proc_chunk['user'].str[0] == 'U']

    # get rid of process ends
    is_proc_start = proc_chunk['start/end'] == 'Start'
    proc_chunk = proc_chunk[is_proc_start]

    print("Only Process starts Len:" + str(len(proc_chunk)))

    # get rid of start/end column, as we only care about processs starts
    proc_chunk = proc_chunk.filter(items=['time', 'user', 'comp', 'process'])

    # append to file
    if i == 0:
        proc_chunk.to_csv("filtered_proc.csv", index=False)
    else:
        proc_chunk.to_csv("filtered_proc.csv", mode='a', header=False, index=False)

    i += 1


