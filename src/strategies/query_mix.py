import pandas as pd 
import numpy as np
from utils.prepare_query import idx_highest_diff_unlabel, idx_lowest_diff_unlabel

def idx_mix_to_label(diff_top_two_pred, labeled_index, n_instances):
    idx_highest = idx_highest_diff_unlabel(diff_top_two_pred, labeled_index)
    idx_lowest = idx_lowest_diff_unlabel(diff_top_two_pred, labeled_index)
    return idx_highest[:n_instances] + idx_lowest[:n_instances]