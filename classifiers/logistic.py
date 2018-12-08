from __future__ import print_function

__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import numpy as np
import pickle
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from classifiers.base import classify

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("input_dir", None,
                    "Dir containing features and labels")
flags.DEFINE_string("output_dir", None,
                    "Dir where results will be saved")
flags.DEFINE_string("features_filename", "human-features.npy",
                    "File containing the numpy array for features")
flags.DEFINE_string("labels_filename", "human-labels.npy",
                    "File containing the numpy array for labels")
flags.DEFINE_integer("val_size", 10,
                     "Number of cross validation runs")


def main(_):
    X = np.load(FLAGS.input_dir + '/' + FLAGS.features_filename)
    y = np.load(FLAGS.input_dir + '/' + FLAGS.labels_filename)

    method = LogisticRegression()
    results = classify(method, X, y, val_iter=FLAGS.val_size)

    output_dir = FLAGS.output_dir or FLAGS.input_dir

    with open('{}/clf_results.pkl'.format(output_dir), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    tf.app.run()
