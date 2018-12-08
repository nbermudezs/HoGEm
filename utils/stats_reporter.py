__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import numpy as np
import tensorflow as tf

from scipy.stats import f_oneway
from utils.summarizer import load_baseline_results, load_results, summarize_replications

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("clf_results_dir", None,
                    "Path to classifier results dir")


def anova_tests(data):
    methods = ['Mashup', 'GraphSAGE']
    result = {}

    keys = {
        'f1_micro': 'f1',
        'accuracy': 'acc',
        'recall_micro': 'recall',
        'precision_micro': 'precision'
    }

    for method in methods:
        baseline = data[method]
        for data_key, fmt_key in keys.items():
            key = '{}_{}'.format(fmt_key, method.lower())
            result[key] = "%.3g" % f_oneway(baseline[data_key], data['HoGEm'][data_key]).pvalue
    return result


def to_latex(all_data):
    template = open('misc/table_template.tex').read().\
        replace('{', '{{').replace('}', '}}').\
        replace('[[', '${').replace(']]', '}$')
    keys = {
        'f1_micro': 'f1',
        'accuracy': 'acc',
        'recall_micro': 'recall',
        'precision_micro': 'precision'
    }
    pos = {
        'f1_micro': 1,
        'accuracy': 2,
        'recall_micro': 3,
        'precision_micro': 4
    }

    format_keys = {}
    for n_classes, data in all_data.items():
        for idx in range(3):
            for data_key, metric_key in keys.items():
                method = data['mean'][idx][0]
                key = '{}_{}_{}'.format(metric_key, method.lower(), n_classes)
                value = '{} \\pm {}'.format(
                    np.round(data['mean'][idx][pos[data_key]], 2),
                    np.round(data['std'][idx][pos[data_key]], 2))
                format_keys[key] = value
    return template.format(**format_keys)


def p_values_to_latex(p_values):
    template = open('misc/p-value_table_template.tex').read(). \
        replace('{', '{{').replace('}', '}}'). \
        replace('[[', '${').replace(']]', '}$')
    return template.format(**p_values)


if __name__ == '__main__':
    data_22 = {
        "Mashup": load_baseline_results("mashup", 22),
        "GraphSAGE": load_baseline_results("graphsage", 22),
        "HoGEm": load_results(FLAGS.clf_results_dir + '/c22'),
    }
    mean_22, std_22 = summarize_replications(data_22)

    data_211 = {
        "Mashup": load_baseline_results("mashup", 211),
        "GraphSAGE": load_baseline_results("graphsage", 211),
        "HoGEm": load_results(FLAGS.clf_results_dir + '/c211'),
    }
    mean_211, std_211 = summarize_replications(data_211)

    all_data = {
        '22': {
            'mean': mean_22,
            'std': std_22
        },
        '211': {
            'mean': mean_211,
            'std': std_211
        }
    }
    with open(FLAGS.clf_results_dir + '/summary.tex', 'w') as f:
        f.write(to_latex(all_data))

    p_values = anova_tests(data_22)
    with open(FLAGS.clf_results_dir + '/p-values-22.tex', 'w') as f:
        f.write(p_values_to_latex(p_values))

    p_values = anova_tests(data_211)
    with open(FLAGS.clf_results_dir + '/p-values-211.tex', 'w') as f:
        f.write(p_values_to_latex(p_values))
