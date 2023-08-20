import numpy as np
import pandas as pd

from utils.prepare_query import idx_highest_diff_unlabel


# create a function to take an array of index and number of instances then return the array of index of the instances to label
# input: array of index, number of instances to label
# output: array of index of the instances to label
def idx_dominate_to_label(diff_top_two_pred, labeled_index, n_instances):
    idx = idx_highest_diff_unlabel(diff_top_two_pred, labeled_index)
    return idx[:n_instances]

