__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

from utils.summarizer import load_baseline_results, load_results, summarize_replications

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("clf_results_dir", None,
                    "Path to classifier results dir")
flags.DEFINE_integer("n_classes", None, "Number of classes in file")


sns.set(style="whitegrid")


def plot_comparison(mean, std, out):
    columns = ['Method', 'F1', 'Accuracy', 'Recall', 'Precision']
    df = pd.DataFrame.from_records(mean, columns=columns, index='Method').astype({
        'F1': float,
        'Accuracy': float,
        'Precision': float,
        'Recall': float
    })
    errors = pd.DataFrame.from_records(std, columns=columns, index='Method').astype({
        'F1': float,
        'Accuracy': float,
        'Precision': float,
        'Recall': float
    })
    ax = df.transpose().plot.bar(yerr=errors.transpose())
    plt.legend(bbox_to_anchor=(0., -0.172, 1., .11),
               loc=3, ncol=3, mode="expand", borderaxespad=0.)
    for item in ax.get_xticklabels():
        item.set_rotation('horizontal')
    plt.title('Overall performance comparison')
    plt.savefig(out, bbox_inches='tight')


if __name__ == '__main__':
    data = {
        "Mashup": load_baseline_results("mashup", FLAGS.n_classes),
        "GraphSAGE": load_baseline_results("graphsage", FLAGS.n_classes),
        "HoGEm": load_results(FLAGS.clf_results_dir),
    }
    mean, std = summarize_replications(data)
    plot_comparison(mean, std,
                    FLAGS.clf_results_dir + '/baseline_comparison.png')