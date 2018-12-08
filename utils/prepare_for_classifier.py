from __future__ import print_function

__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.joblib import dump


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("embeddings_filepath", None,
                    "Path to the embeddings file to be used as features")
flags.DEFINE_string("labels_filepath", None,
                    "Labels (as adjacency list) to be attached to features")
flags.DEFINE_string("output_dir", None,
                    "Directory where resulting files will be kept")
flags.DEFINE_string("organism", "human",
                    "Which organism these features/labels belong to")


def main(_):
    feats = np.load(FLAGS.embeddings_filepath, encoding='latin1')
    print('Feature matrix shape: ', feats.shape)

    y = pd.read_csv(FLAGS.labels_filepath,
                    header=None, names=['gene', 'mf'], sep='\t')
    print('# links between genes and MFs: ', y.values.shape[0])

    label_encoder = LabelEncoder()
    label_encoder.fit(y.values[:, 1])
    label_size = len(label_encoder.classes_)

    new_features = []
    new_labels = []
    genes = []
    for gene, mfs in y.groupby('gene'):
        genes.append(gene)
        gene_feats = feats[gene - 1, :]
        gene_y = np.zeros(label_size)
        gene_y[label_encoder.transform(mfs['mf'].values)] = 1
        new_features.append(gene_feats)
        new_labels.append(gene_y)

    new_features = np.vstack(new_features)
    new_labels = np.vstack(new_labels)

    print('Shape of selected gene matrix: ', new_features.shape)
    print('Shape of selected label matrix: ', new_labels.shape)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    new_features.dump('{}/{}-features.npy'.format(FLAGS.output_dir, FLAGS.organism))
    new_labels.dump('{}/{}-labels.npy'.format(FLAGS.output_dir, FLAGS.organism))

    np.array(genes).dump('{}/{}-genes.npy'.format(FLAGS.output_dir, FLAGS.organism))
    dump(label_encoder, '{}/{}-labelencoder.pkl'.format(FLAGS.output_dir, FLAGS.organism))


if __name__ == '__main__':
    tf.app.run()
