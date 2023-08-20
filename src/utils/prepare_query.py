import pandas as pd
import numpy as np
# create a function that takes an 2D-array and return the array of top 2 predicted probabilities
# input: 2D-array of predicted probabilities
# output: array of top 2 predicted probabilities
def top_two_pred(pred_proba):
    top_two_pred = []
    for i in range(len(pred_proba)):
        top_two_pred.append(sorted(pred_proba[i], reverse=True)[:2])
    return top_two_pred

# create a function that take an array of top 2 predicted probabilities and return the array of difference between the top 2 predicted probabilities
# input: array of top 2 predicted probabilities
# output: array of difference between the top 2 predicted probabilities
def diff_top_two_pred(top_two_pred):
    diff_top_two_pred = []
    for i in range(len(top_two_pred)):
        diff_top_two_pred.append(top_two_pred[i][0] - top_two_pred[i][1])
    return diff_top_two_pred

# create a function that takes an array of top 2 predicted probabilities and return the array of quotient between the top 2 predicted probabilities
# input: array of top 2 predicted probabilities
# output: array of quotient between the top 2 predicted probabilities
def quot_top_two_pred(top_two_pred):
    quot_top_two_pred = []
    for i in range(len(top_two_pred)):
        quot_top_two_pred.append(top_two_pred[i][0] / top_two_pred[i][1])
    return quot_top_two_pred

# input: array of difference between the top 2 predicted probabilities, list of labeled index
# output: list of index of difference between the top 2 predicted probabilities of the unlabeled instances in descending order
def idx_highest_diff_unlabel(diff_top_two_pred, labeled_index):
    diff_top_two_pred = pd.Series(diff_top_two_pred)
    mask = ~diff_top_two_pred.index.isin(labeled_index)
    sliced_diff_top_two_pred = diff_top_two_pred[mask]
    sliced_diff_top_two_pred = sliced_diff_top_two_pred.sort_values(ascending=False)

    return np.array(sliced_diff_top_two_pred.index)

def idx_lowest_diff_unlabel(diff_top_two_pred, labeled_index):
    diff_top_two_pred = pd.Series(diff_top_two_pred)
    mask = ~diff_top_two_pred.index.isin(labeled_index)
    sliced_diff_top_two_pred = diff_top_two_pred[mask]
    sliced_diff_top_two_pred = sliced_diff_top_two_pred.sort_values(ascending=True)

    return np.array(sliced_diff_top_two_pred.index)



