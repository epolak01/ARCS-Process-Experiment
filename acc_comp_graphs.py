####################################
# Author: Emil Polakiewicz
# Date: December 2021
# Purpose: Graph accuracies of linear regression and random forest classfiers
####################################

#####################
# Imports
#####################

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


#####################
# Globals
#####################

BASE_DIR = os.getcwd()
LR_DATA_DIR = os.path.join(BASE_DIR, '/Final Results/lr_acc')
RF_DATA_DIR = os.path.join(BASE_DIR, '/Final Results/rf_acc')

#####################
# Functions
#####################

def load_local_file(filename, DIR):
    try:
        data = pd.read_csv(os.path.join(DIR, filename), delimiter=",", header=None).values
    except pd.errors.EmptyDataError:
        data = np.zeros((0, 0))
    return data

def get_feature_set_name(num):
    if num == 0:
        return 'Proc + Session Features'
    elif num == 1:
        return 'Proc Features'
    elif num == 2:
        return 'Proc Name'

def graph_accuracy_comparisons(acc_lists, clf_name):
    #print(acc_lists)
    acc_list_len = len(acc_lists[0])
    plt.figure(0)
    for j in range(acc_list_len):
        acc1 = acc_lists[:, j]
        for k in range(acc_list_len):
            #if j != k:
            acc2 = acc_lists[:, k]
            ax = plt.subplot2grid((acc_list_len, acc_list_len), (j, k))
            #ax.title.set_text(get_feature_set_name(j) + " vs. " + get_feature_set_name(k))
            ax.scatter(acc1, acc2)
            lims = [0, 1]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
            ax.set_xlim([0, 1])
            ax.set_xlim([0, 1])

            # if j == 0 and k == 1:
            #     ax.set_ylabel('All Feature Accuracy', fontsize="xx-small")
            # if j != 0 and k == 0:
            #     ax.set_ylabel(get_feature_set_name(j) + " Accuracy", fontsize="xx-small")
            if k == 0 and j != acc_list_len - 1:
                ax.set_ylabel(get_feature_set_name(j) + " Accuracy", fontsize="xx-small")
                ax.tick_params(axis='x', which='both', labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            if j == acc_list_len - 1 and k != 0:
                ax.set_xlabel(get_feature_set_name(k) + " Accuracy", fontsize="xx-small")
                ax.tick_params(axis='y', which='both', labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            if k != 0 and j != acc_list_len - 1:
                ax.tick_params(axis='x', which='both', labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                ax.tick_params(axis='y', which='both', labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            if k == 0 and j == acc_list_len - 1:
                ax.set_ylabel(get_feature_set_name(j) + " Accuracy", fontsize="xx-small")
                ax.set_xlabel(get_feature_set_name(k) + " Accuracy", fontsize="xx-small")

    plt.savefig("/Final Results/3x3 graphs/3x3_" + clf_name +".png", bbox_inches='tight')
    plt.close()

#####################
# Graphs
#####################

# Load files
lr_proc_sess_acc = load_local_file('lr_proc_all_acc_te.csv', LR_DATA_DIR)
lr_proc_acc = load_local_file('lr_proc_acc_te.csv', LR_DATA_DIR)
lr_proc_name_acc = load_local_file('lr_proc_name_acc_te.csv', LR_DATA_DIR)
lr_proc_n_test_acc_list = np.column_stack((lr_proc_sess_acc, lr_proc_acc, lr_proc_name_acc))

rf_proc_sess_acc = load_local_file('rf_proc_all_acc_te.csv', RF_DATA_DIR)
rf_proc_acc = load_local_file('rf_proc_acc_te.csv', RF_DATA_DIR)
rf_proc_name_acc = load_local_file('rf_proc_name_acc_te.csv', RF_DATA_DIR)
rf_proc_n_test_acc_list = np.column_stack((rf_proc_sess_acc, rf_proc_acc, rf_proc_name_acc))

# graph accuracies
graph_accuracy_comparisons(lr_proc_n_test_acc_list, "lr")
graph_accuracy_comparisons(rf_proc_n_test_acc_list, "rf")

# Create Boxplots
lr_acc_df = pd.DataFrame(lr_proc_n_test_acc_list, columns=['Session + Proc Features', 'Proc Features', 'Proc Name'])
rf_acc_df = pd.DataFrame(rf_proc_n_test_acc_list, columns=['Session + Proc Features', 'Proc Features', 'Proc Name'])

lr_boxplot = lr_acc_df.boxplot()
lr_boxplot.plot()
plt.ylabel("Accuracy")
plt.xlabel("Feature Combinations")
plt.ylim([0, 1])
plt.title("Boxplot of Accuracy for Feature Combinations in Logistic Regression Models")
plt.savefig("/Final Results/box plots/lr_boxplot.png", bbox_inches='tight')
plt.close()

rf_boxplot = rf_acc_df.boxplot()
rf_boxplot.plot()
plt.ylabel("Accuracy")
plt.xlabel("Feature Combinations")
plt.ylim([.3, 1])
plt.title("Boxplot of Accuracy for Feature Combinations")
plt.savefig("/Final Results/box plots/rf_boxplot.png", bbox_inches='tight')
plt.close()

