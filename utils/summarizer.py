__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import numpy as np
import pickle


def load_baseline_results(method, n_classes):
    with open('baseline_results/c{}/{}.pkl'.format(n_classes, method), 'rb') as f:
        return pickle.load(f)


def load_results(dir):
    with open('{}/clf_results.pkl'.format(dir), 'rb') as f:
        return pickle.load(f)


def summarize_replications(data_dict):
    result = []
    errors = []
    for method in ['Mashup', 'GraphSAGE', 'HoGEm']:
        data = data_dict[method]
        result.append([
            method,
            np.mean(data['f1_micro']),
            np.mean(data['accuracy']),
            np.mean(data['recall_micro']),
            np.mean(data['precision_micro'])])
        errors.append([
            method,
            np.std(data['f1_micro']),
            np.std(data['accuracy']),
            np.std(data['recall_micro']),
            np.std(data['precision_micro'])])
    return result, errors
