####################################
# Author: Emil Polakiewicz
# Date: December 2021
# Purpose: Finish Preprocessing and run logistic regression and random forest classifiers on data
####################################


#####################
# Imports
#####################

import numpy as np
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics
import scipy
import random
from sklearn.preprocessing import MaxAbsScaler

#####################
# Globals
#####################

num_features = 17
num_prev_proc = 5
time_step = 3600
num_proc = 50
percent_activity_threshold = 1
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


# Converts first column of data from user_nums, to labels given a target user
def label_dataset(user_num, y):
    return y.apply(lambda x: 1 if x == user_num else 0)

def get_feature_set_name(num):
    if num == 0:
        return 'Proc + Session Features'
    elif num == 1:
        return 'Proc Features'
    elif num == 2:
        return 'Proc Name'


# print a table of metrics on data
def metrics_table(y_tr_N, y_te_N, y_va_N):
    # initialize list of lists
    data = [['ex', len(y_tr_N), len(y_te_N), len(y_va_N)],
            ['p ex', sum(y_tr_N), sum(y_te_N), sum(y_va_N)]]

    new_data = ['frac of p ex']
    for i in range(1, 4):
        if i == 2:
            new_data.append("N/a")
        else:
            if data[0][i]:
                new_data.append(data[1][i] / data[0][i])
            else:
                new_data.append(0)
    data.append(new_data)

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['Stat', 'Training', 'Testing', "Validation"])
    return df

# run grid search on single user
def logistic_regression_search(C_grid, x_train, x_validation, x_test, y_train, y_validation, y_test):
    log_los_tr_C_list = []
    log_los_va_C_list = []
    acc_va_C_list = []

    # Build and evaluate model for each value C
    for C in C_grid:
        # create model
        clf = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, C=C, class_weight="balanced")

        # train model
        clf.fit(x_train, y_train)
        proba_tr = clf.predict_proba(x_train)
        proba_va = clf.predict_proba(x_validation)

        # record accuracy
        acc_va_C_list.append(clf.score(x_validation, y_validation))

        # record log loss
        log_loss = sklearn.metrics.log_loss(y_train, proba_tr, labels=[0, 1])
        log_los_tr_C_list.append(log_loss)
        log_loss = sklearn.metrics.log_loss(y_validation, proba_va, labels=[0, 1])
        log_los_va_C_list.append(log_loss)

    # find minimum value based on log loss
    best_C_ind = log_los_va_C_list.index(min(log_los_va_C_list))

    # record best values
    best_C_va = C_grid[best_C_ind]
    best_log_los_va = log_los_va_C_list[best_C_ind]
    best_acc_va = acc_va_C_list[best_C_ind]

    #######
    # run best model on test set
    ######
    # create model
    clf = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, C=best_C_va,
                                                  class_weight="balanced")
    # train model
    clf.fit(x_train, y_train)
    proba_va = clf.predict_proba(x_validation)[:, 1]
    proba_te = clf.predict_proba(x_test)[:, 1]
    acc_te = clf.score(x_test, y_test)
    log_loss_te = sklearn.metrics.log_loss(y_test, proba_te, labels=[0, 1])
    a_roc_va = sklearn.metrics.roc_auc_score(y_validation, proba_va)
    a_roc_te = sklearn.metrics.roc_auc_score(y_test, proba_te)

    return best_C_va, log_los_tr_C_list, log_los_va_C_list, best_log_los_va, best_acc_va, acc_te, log_loss_te,\
           a_roc_va, a_roc_te, proba_va

# run grid search on single user
def random_forest_search(max_depths, min_leafs, x_train, x_validation, x_test, y_train, y_validation, y_test):
    tr_log_loss = []
    va_log_loss = []
    va_acc = []

    # Build and evaluate model for each value C
    for max_depth in max_depths:
        log_los_tr_min_leaf_list = []
        log_los_va_min_leaf_list = []
        acc_va_min_leaf_list = []

        for min_leaf in min_leafs:
            # create model
            clf = sklearn.tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2, class_weight='balanced',
                                                      max_depth=max_depth, min_samples_leaf=min_leaf)

            # train model
            clf.fit(x_train, y_train)
            proba_tr = clf.predict_proba(x_train)[:, 1]
            proba_va = clf.predict_proba(x_validation)[:, 1]

            # record accuracy
            acc_va_min_leaf_list.append(clf.score(x_validation, y_validation))

            # record log loss
            log_loss = sklearn.metrics.log_loss(y_train, proba_tr, labels=[0, 1])
            log_los_tr_min_leaf_list.append(log_loss)
            log_loss = sklearn.metrics.log_loss(y_validation, proba_va, labels=[0, 1])
            log_los_va_min_leaf_list.append(log_loss)

        tr_log_loss.append(log_los_tr_min_leaf_list)
        va_log_loss.append(log_los_va_min_leaf_list)
        va_acc.append(acc_va_min_leaf_list)

    tr_log_loss = np.array(tr_log_loss)
    va_log_loss = np.array(va_log_loss)
    va_acc = np.array(va_acc)

    min_log_loss_tr_ind =  np.unravel_index(np.argmin(tr_log_loss), tr_log_loss.shape)
    min_log_loss_va_ind =  np.unravel_index(np.argmin(va_log_loss), va_log_loss.shape)
    max_acc_va_ind =  np.unravel_index(np.argmax(va_acc), va_acc.shape)

    min_log_loss_tr = tr_log_loss[min_log_loss_tr_ind[0], min_log_loss_tr_ind[1]]
    min_log_loss_va = va_log_loss[min_log_loss_va_ind[0]][min_log_loss_va_ind[1]]
    max_acc_va = va_acc[max_acc_va_ind[0]][max_acc_va_ind[1]]

    best_max_depth_va = max_depths[min_log_loss_va_ind[0]]
    best_min_leaf_va = min_leafs[min_log_loss_va_ind[1]]

    #######
    # run best model on test set
    ######
    # create model
    clf = sklearn.tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2, class_weight='balanced',
                                              max_depth=best_max_depth_va, min_samples_leaf=best_min_leaf_va)
    # train model
    clf.fit(x_train, y_train)
    proba_va = clf.predict_proba(x_validation)[:, 1]
    proba_te = clf.predict_proba(x_test)[:, 1]
    acc_te = clf.score(x_test, y_test)
    log_loss_te = sklearn.metrics.log_loss(y_test, proba_te, labels=[0, 1])
    a_roc_va = sklearn.metrics.roc_auc_score(y_validation, proba_va)
    a_roc_te = sklearn.metrics.roc_auc_score(y_test, proba_te)

    return best_max_depth_va, best_min_leaf_va, min_log_loss_va, max_acc_va, acc_te, log_loss_te,\
           a_roc_va, a_roc_te, proba_va

# run grid search on multiple users, and make graphs
def run_logistic_regression_users(target_users, data_name, x_train, x_validation, x_test, y_train, y_validation, y_test,
                                  graph, metrics, print_results):
    va_acc_list = []
    test_acc_list = []
    va_roc_list = []
    te_roc_list = []

    for user_num in target_users:
        y_tr_labled = label_dataset(user_num, y_train)
        y_va_labled = label_dataset(user_num, y_validation)
        y_te_labled = label_dataset(user_num, y_test)

        if (sum(y_tr_labled) < 10) or (sum(y_va_labled) < 5) or (sum(y_te_labled) < 5):
            continue

        if metrics:
            print("Metrics for User: " + str(user_num))
            print(metrics_table(y_tr_labled, y_te_labled, y_va_labled))

        print("Running LR on User")
        C_grid = np.logspace(-6, 15, 4)
        best_C, tr_log_los_list, va_log_loss_list, va_best_log_loss, va_best_acc, te_acc, te_log_loss, va_roc, te_roc, \
            proba_va = logistic_regression_search(C_grid, x_train, x_validation, x_test, y_tr_labled,
                                                  y_va_labled, y_te_labled)

        va_acc_list.append(va_best_acc)
        test_acc_list.append(te_acc)
        va_roc_list.append(va_roc)
        te_roc_list.append(te_roc)

        print("Graphing User")
        if graph:
            # create plot for train data across C
            plt.xlabel('log(C)');
            plt.ylabel('Log Loss');
            plt.xscale('log')
            plt.ylim(0, 1)
            plt.plot(C_grid, tr_log_los_list, 'bs-', label='Training Log Loss');
            plt.plot(C_grid, va_log_loss_list, 'rs-', label='Validation Log Loss');
            plt.legend(loc='best');
            plt.title(
                'User: ' + str(user_num) + ' Logistic Regression ' + data_name + ' C Analysis: Log Loss vs. C');
            plt.savefig("log_C_" + str(user_num) + "_" + data_name + ".png")
            plt.close()

            # roc graph
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_va_labled, proba_va)
            plt.plot(fpr, tpr, ls='-', label=data_name)

            # Finish up ROC curve graph
            plt.title("User " + str(user_num) + " ROC on Validation Set Logistic Regression");
            plt.xlabel('false positive rate');
            plt.ylabel('true positive rate');
            plt.legend(loc='lower right', fontsize=10, bbox_to_anchor=(1.5, 0.5));
            B = 0.01
            plt.xlim([0 - B, 1 + B]);
            plt.ylim([0 - B, 1 + B]);
            plt.savefig("user_" + str(user_num) + "_" + data_name + "roc_graph.png", bbox_inches='tight')
            plt.close()

        if print_results:
            # print results
            print("User: " + str(user_num) + " Best C on VAL : %.3f" % (best_C))
            print("User: " + str(user_num) + " Best Log Loss on VAL : %.3f" % (va_best_log_loss))
            print("User: " + str(user_num) + " Best Acc on VAL : %.3f" % (va_best_acc))
            print("User: " + str(user_num) + " Best Log Loss on TEST : %.3f" % (te_log_loss))
            print("User: " + str(user_num) + " Best Acc on TEST : %.3f" % (te_acc))

    return va_acc_list, test_acc_list, va_roc_list, te_roc_list

# run grid search on multiple users, and make graphs
def run_random_forest_users(target_users, data_name, x_train, x_validation, x_test, y_train, y_validation, y_test, metrics, graph,
                                  print_results):
    va_acc_list = []
    test_acc_list = []
    va_roc_list = []
    te_roc_list = []

    for user_num in target_users:
        y_tr_labled = label_dataset(user_num, y_train)
        y_va_labled = label_dataset(user_num, y_validation)
        y_te_labled = label_dataset(user_num, y_test)

        if (sum(y_tr_labled) < 10) or (sum(y_va_labled) < 5) or (sum(y_te_labled) < 5):
            continue

        if metrics:
            print("Metrics for User: " + str(user_num))
            print(metrics_table(y_tr_labled, y_te_labled, y_va_labled))

        max_depths = [8, 128]
        min_leafs = [1, 3]
        best_max_depth, best_min_leaf, va_best_log_loss, va_best_acc, te_acc, te_log_loss, va_roc, te_roc, proba_va = \
            random_forest_search(max_depths, min_leafs, x_train, x_validation, x_test, y_tr_labled, y_va_labled,
                                 y_te_labled)

        va_acc_list.append(va_best_acc)
        test_acc_list.append(te_acc)
        va_roc_list.append(va_roc)
        te_roc_list.append(te_roc)


        if graph:
            # roc graph
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_va_labled, proba_va)
            plt.plot(fpr, tpr, ls='-', label=data_name)

            # Finish up ROC curve graph
            plt.title("User " + str(user_num) + " ROC on Validation Set Random Forest");
            plt.xlabel('false positive rate');
            plt.ylabel('true positive rate');
            plt.legend(loc='lower right', fontsize=10, bbox_to_anchor=(1.5, 0.5));
            B = 0.01
            plt.xlim([0 - B, 1 + B]);
            plt.ylim([0 - B, 1 + B]);
            plt.savefig("user_" + str(user_num) + "_" + data_name + "roc_graph.png", bbox_inches='tight')
            plt.close()

        if print_results:
            # print results
            print("User: " + str(user_num) + " Best max_depth on VAL : %.3f" % (best_max_depth))
            print("User: " + str(user_num) + " Best best_min_leaf on VAL : %.3f" % (best_min_leaf))
            print("User: " + str(user_num) + " Best Log Loss on VAL : %.3f" % (va_best_log_loss))
            print("User: " + str(user_num) + " Best Acc on VAL : %.3f" % (va_best_acc))
            print("User: " + str(user_num) + " Best Log Loss on TEST : %.3f" % (te_log_loss))
            print("User: " + str(user_num) + " Best Acc on TEST : %.3f" % (te_acc))

    return va_acc_list, test_acc_list, va_roc_list, te_roc_list

#########################
# Load Data
#########################

x = load_local_file("all_proc_data.csv")

# loading popular processess to one hot encode
process_info = load_local_file("final_process_analysis.csv").values
processes = process_info[:, 1]
processes = processes[:num_proc]

temp_list = [-1]
num_proc = num_proc + 1

# convert process names to ints
for i in range(len(processes)):
    temp_str = processes[i]
    temp_list.append(int(temp_str[1:]))

processes = np.array(temp_list)
processes.sort()

#########################
# Define Splits for Data
#########################

# Define training, validation, and testing splits
# Training Set: Days 30-58
#               Seconds: from 2592000 until end
tr_start_time = 2592000

# Testing Set: Days 16-29
#              Seconds: from 1296001 to 2591999
te_start_time = 1296000
te_end_time = 2592000

# Validation Set: Days 0-15
#                 Seconds: from 0 to 1296000
va_end_time = 1296000

# Find indices for splits in dataset
tr_start_index = int(np.searchsorted(x.iloc[:, 1], tr_start_time, side='right') + 1)
te_start_index = int(np.searchsorted(x.iloc[:, 1], te_start_time, side='right') + 1)
te_end_index = int(np.searchsorted(x.iloc[:, 1], te_end_time, side='right') + 1)
va_end_index = int(np.searchsorted(x.iloc[:, 1], va_end_time, side='right') + 1)

y = x.iloc[:,0]
x = x.iloc[:,1:]

#########################
# One hot encode data
#########################

# Encoding other category by only recording info for processses we recognize
processes_set = set(processes)
x['Process Name'] = x['Process Name'].map(lambda e: -1 if e not in processes_set else e)
x['5th Prev Process'] = x['5th Prev Process'].map(lambda e: -1 if e not in processes_set else e)
x['4th Prev Process'] = x['4th Prev Process'].map(lambda e: -1 if e not in processes_set else e)
x[ '3rd Prev Process'] = x['3rd Prev Process'].map(lambda e: -1 if e not in processes_set else e)
x['2nd Prev Process'] = x['2nd Prev Process'].map(lambda e: -1 if e not in processes_set else e)
x['1st Prev Process'] = x['1st Prev Process'].map(lambda e: -1 if e not in processes_set else e)

one_hot_encoder = OneHotEncoder(sparse=True, categories=[processes, processes, processes, processes, processes, processes], handle_unknown='ignore')

# the reshape is only cuz it has one column, when you have more, won't need this
x_one_hot_proc_name = one_hot_encoder.fit_transform(x[['Process Name', '5th Prev Process', '4th Prev Process',
                                                      '3rd Prev Process', '2nd Prev Process', '1st Prev Process']].values)
x_one_hot_proc_name_df = pd.DataFrame(x_one_hot_proc_name)
x.drop(columns=['Process Name', '5th Prev Process', '4th Prev Process', '3rd Prev Process', '2nd Prev Process', '1st Prev Process'], axis=1)
x = scipy.sparse.hstack([scipy.sparse.csr_matrix(x.values), x_one_hot_proc_name]).tocsr()

#########################
# Min Max Scale data
#########################

transformer = MaxAbsScaler(copy=False)
transformer.fit_transform(x)
x = x.tocsr()

#########################
# Split Data
#########################

# split data into training, validation, testing
x_tr = x[tr_start_index:, :]
y_tr = y.iloc[tr_start_index:]

x_va = x[:va_end_index, :]
y_va = y.iloc[:va_end_index]

x_te = x[te_start_index:te_end_index, :]
y_te = y.iloc[te_start_index:te_end_index]

#########################
# Get random users
#########################

random.seed(0)
target_users = random.sample(range(1, 5000), 10)

#########################
# Linear Regression/Random Forest Classifier On Process and Session Features
#########################

# run models on target users to get individual user graphs and statistics
lr_proc_n_va_acc_list, lr_proc_n_test_acc_list, lr_proc_n_va_roc_list, lr_proc_n_te_roc_list = \
    run_logistic_regression_users(target_users, "Session + Process Features", x_tr, x_va, x_te, y_tr, y_va,
                                  y_te, True, True, True)

np.savetxt("lr_proc_all_roc_va.csv", np.array(lr_proc_n_va_roc_list), delimiter=',')
np.savetxt("lr_proc_all_roc_te.csv", np.array(lr_proc_n_te_roc_list), delimiter=',')
np.savetxt("lr_proc_all_acc_va.csv", np.array(lr_proc_n_va_acc_list), delimiter=',')
np.savetxt("lr_proc_all_acc_te.csv", np.array(lr_proc_n_test_acc_list), delimiter=',')

rf_proc_n_va_acc_list, rf_proc_n_test_acc_list, rf_proc_n_va_roc_list, rf_proc_n_te_roc_list = \
    run_random_forest_users(target_users, "Session + Process Features", x_tr, x_va, x_te, y_tr, y_va, y_te, True, True, True)

np.savetxt("rf_proc_all_roc_va.csv", np.array(rf_proc_n_va_roc_list), delimiter=',')
np.savetxt("rf_proc_all_roc_te.csv", np.array(rf_proc_n_te_roc_list), delimiter=',')
np.savetxt("rf_proc_all_acc_va.csv", np.array(rf_proc_n_va_acc_list), delimiter=',')
np.savetxt("rf_proc_all_acc_te.csv", np.array(rf_proc_n_test_acc_list), delimiter=',')