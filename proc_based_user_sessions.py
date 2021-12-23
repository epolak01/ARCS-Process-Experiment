####################################
# Author: Emil Polakiewicz
# Date: December 2021
# Purpose: Pre-process proc_data into processs and session features
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

num_features = 17
num_prev_proc = 5
time_step = 3600
num_proc = 50
percent_activity_threshold = 1
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../../filt_data')


#####################
# Data functions
#####################

def loadfile(filename):
    if not os.path.exists(os.path.join(DATA_DIR, filename)):
        return np.zeros((0, 10))

    try:
        data = pd.read_csv(os.path.join(DATA_DIR, filename), delimiter=",", nrows=100000)
    except pd.errors.EmptyDataError:
        data = np.zeros((0, 10))
    return data


def load_local_file(filename):
    try:
        data = pd.read_csv(os.path.join(BASE_DIR, filename), delimiter=",").values
    except pd.errors.EmptyDataError:
        data = np.zeros((0, 0))
    return data


# returns true if value is in range (start, end)
def in_range(start, end, val):
    return (val >= start) and (val <= end)

# calculates average session length
def ave_session_length(sessions):
    result = 0
    min_sess = 10 ** 10
    max_sess = 0

    if len(sessions) == 0:
        return 0,0,0

    for i in range(len(sessions)):
        curr = sessions[i][1] - sessions[i][0]
        result += curr
        max_sess = max(curr, max_sess)
        min_sess = min(curr, min_sess)
    return min_sess, max_sess, result // len(sessions)

# Sorting based on the increasing order of the start intervals
def mergeIntervals(arr):
    # 'm' array contains the list of all merged intervals
    m = []
    # 's' gives the starting point of that interval
    s = -10000
    # 'max' value gives the last point of that particular interval
    max = -100000
    for i in range(len(arr)):
        a = arr[i]
        if a[0] > max:
            if i != 0:
                m.append([s, max])
            max = a[1]
            s = a[0]
        else:
            if a[1] >= max:
                max = a[1]

    if max != -100000 and [s, max] not in m:
        m.append([s, max])
    return m

# given process data for a single user, calculates user sessions
def get_sessions(proc_data):
    sessions = []
    sessions_inds = []
    activity_during_session = []

    time_session_start = 0
    ind_session_start = 0
    in_session = False
    length = len(proc_data)

    for i in range(length):
        # when we are at the last session, must terminate current session
        if i == (length - 1):
            # when we are not in a session, create a new one
            if not in_session:
                time_session_start = proc_data[i][0]
                ind_session_start = i

            # end session
            time_session_end = proc_data[i][0] + time_step
            ind_session_end = i
            sessions.append((time_session_start, time_session_end))
            sessions_inds.append((ind_session_start, ind_session_end))
            activity_during_session.append(ind_session_end - ind_session_start + 1)
            in_session = False
        else:
            # if we are not in a session start a new session
            if not in_session:
                time_session_start = proc_data[i][0]
                ind_session_start = i
                in_session = True

            # we are currently in a session
            # next process is not in timestep, end session
            elif proc_data[i + 1][0] > (proc_data[i][0] + time_step):
                time_session_end = proc_data[i][0] + time_step
                ind_session_end = i
                sessions.append((time_session_start, time_session_end))
                sessions_inds.append((ind_session_start, ind_session_end))
                activity_during_session.append(ind_session_end - ind_session_start + 1)
                in_session = False

    return sessions, activity_during_session


# returns the index of the session the time belongs to
def get_session(sessions, time):
    for i in range(len(sessions)):
        if in_range(sessions[i][0], sessions[i][1], time):
            return i
    return -1


# returns 'Time Since Session Start', 'Time Until Session End', 'Session Length', 'Session Start Time of Day',
# 'Session End Time of Day, 'Session Time of Week' for a particular process
def get_session_times(sessions, proc_time):
    for session in sessions:
        if in_range(session[0], session[1], proc_time):
            return proc_time - session[0], session[1] - proc_time, session[1] - session[0],\
                   (session[0] // 3600) % 24, (session[1] // 3600) % 24, (session[0] // 86400) % 7


# returns the process features given process starts, previous processes and user num
def get_proc_feature(proc, sessions, prev_procs, user_num):
    proc_features = np.zeros(num_features)
    currlen = 0

    # user num
    proc_features[currlen] = user_num
    currlen += 1

    # time
    proc_features[currlen] = proc[0]
    currlen += 1

    # Time of day (hour 0-23)
    proc_features[currlen] = (proc[0] // 3600) % 24
    currlen += 1

    # Time of the Week (day 0-6)
    proc_features[currlen] = (proc[0] // 86400) % 7
    currlen += 1

    # session_id
    session_id = get_session(sessions, proc[0])
    proc_features[currlen] = session_id + 1
    currlen += 1

    since_start, until_end, sess_len, start_tod, end_tod, dow = get_session_times(sessions, proc[0])

    # time elapsed since session start
    if session_id == -1:
        proc_features[currlen] = 0
    else:
        proc_features[currlen] = since_start
    currlen += 1

    # time until session end
    if session_id == -1:
        proc_features[currlen] = 0
    else:
        proc_features[currlen] = until_end
    currlen += 1

    # session length
    if session_id == -1:
        proc_features[currlen] = 0
    else:
        proc_features[currlen] = sess_len
    currlen += 1

    # session start time of day
    if session_id == -1:
        proc_features[currlen] = 0
    else:
        proc_features[currlen] = start_tod
    currlen += 1

    # session end time of day
    if session_id == -1:
        proc_features[currlen] = 0
    else:
        proc_features[currlen] = end_tod
    currlen += 1

    # session day of week
    if session_id == -1:
        proc_features[currlen] = 0
    else:
        proc_features[currlen] = dow
    currlen += 1

    # process name
    temp_str = proc[3]
    proc_features[currlen] = int(temp_str[1:])
    currlen += 1

    # prev procs
    for i in range(len(prev_procs)):
        proc_features[currlen] = prev_procs[i]
        currlen += 1

    return proc_features

# retrieve features for all processes in data
def get_all_proc_features(procs, sessions, user_num):
    features = []
    for i in range(len(procs)):
        # retrieve previous processes
        prev_procs = []
        if i > num_prev_proc:
            prev_procs = procs[i - num_prev_proc: i]
            prev_procs = prev_procs[:, 3]
        else:
            for j in range(num_prev_proc - i):
                prev_procs.append("P0")
            # loop backwards
            for j in range(i - 1, -1 , -1):
                prev_procs.append(procs[i - (j + 1)][3])

        # turn previous processes names to ints
        for k in range(len(prev_procs)):
            temp_str = prev_procs[k]
            if not isinstance(temp_str, int):
                prev_procs[k] = int(temp_str[1:])

        # get features for particular processs
        features.append(get_proc_feature(procs[i], sessions, prev_procs, user_num))
    return features

############
# Main
############

# load process data file
all_proc_data = loadfile("filtered_proc.csv")

# sort process data by user
all_proc_data.sort_values(by=['user'], inplace=True)

# get number of processes run by each user
user_counts = all_proc_data['user'].value_counts().to_frame('count')
user_counts['user'] = user_counts.index
user_counts.reset_index(drop=True, inplace=True)

# sort process counts such that they align with sorted process data
user_counts.sort_values(by=['user'], inplace=True)

sessions_per_user = []
processes_per_session = []
unique_proc_per_user = []
user_data = []

all_proc_ind = 0
i = 0
# iterate through users using indices generated by user counts
for count, user in user_counts.values:
    proc_data = all_proc_data.iloc[all_proc_ind: all_proc_ind + count].values
    all_proc_ind += count

    # sort proc data by time
    proc_data = proc_data[np.argsort(proc_data[:, 0])]

    # Get sessions
    sessions, session_activity = get_sessions(proc_data)
    sessions = mergeIntervals(sessions)

    # save session/user statistics
    sessions_per_user.append(len(sessions))
    processes_per_session.append(sum(session_activity))
    unique_proc_per_user.append(len(np.unique(proc_data[:, 3])))

    # get proc features for dataset
    data = get_all_proc_features(proc_data, sessions, i)

    # convert features into dataframe
    df = pd.DataFrame(data, columns=['User Num', 'Time', 'Time of day', 'Day of the Week', 'Session_id', 'Time Since Session Start',
                                     'Time Until Session End', 'Session Length', 'Session Start Time of Day',
                                     'Session End Time of Day', 'Session Time of Week', 'Process Name',
                                     '5th Prev Process', '4th Prev Process', '3rd Prev Process', '2nd Prev Process',
                                     '1st Prev Process'])

    user_data.append(df)
    i += 1

# save some user statistics for further analysis
sessions_per_user_df = pd.DataFrame(sessions_per_user, columns=["num_sessions"])
sessions_per_user_df.to_csv("sessions_per_user.csv", index=True)
processes_per_session_df = pd.DataFrame(processes_per_session, columns=["total_processes"])
processes_per_session_df.to_csv("processes_per_session.csv", index=True)
user_counts.to_csv("processes_per_user.csv", index=True)
unique_proc_per_user_df = pd.DataFrame(unique_proc_per_user, columns=["unique_processes"])
unique_proc_per_user_df.to_csv("unique_processes_per_user.csv", index=True)

# Concatenate dataset into one dataframe
all_data = pd.concat(user_data)

# sort data by time
all_data.sort_values(by=['Time'], inplace=True)

# write data to file
all_data.to_csv("all_proc_data.csv", index=False)
