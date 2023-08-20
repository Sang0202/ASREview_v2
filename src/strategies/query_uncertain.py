import numpy as np
import pandas as pd

from utils.prepare_query import idx_lowest_diff_unlabel

def idx_dominate_to_label(diff_top_two_pred, labeled_index, n_instances):
    idx = idx_lowest_diff_unlabel(diff_top_two_pred, labeled_index)
    return idx[:n_instances]


