from __future__ import print_function

__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances as ped

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", None,
                    "Path containing the raw gene embeddings")
flags.DEFINE_string("homologs_filepath", None,
                    "Path to fole containing the homolog data")


OFFSET = 1000000


def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2.))


def get_homolog_distances(human_embs, yeast_embs):
    return np.sqrt(np.sum((human_embs - yeast_embs) ** 2., axis=1))


if __name__ == '__main__':
    human_embeddings = np.load(FLAGS.input_dir + '/embs_human-tSNE_2D.npy')
    yeast_embeddings = np.load(FLAGS.input_dir + '/embs_yeast-tSNE_2D.npy')

    # human_distances = ped(human_embeddings).flatten()
    # print('All distances')
    # print('Mean: ', np.mean(human_distances))
    # print('Std: ', np.std(human_distances))

    homologs = pd.read_csv(FLAGS.homologs_filepath, sep='\t', names=['human', 'yeast'])
    homologs['yeast'] = homologs['yeast'] - OFFSET

    homolog_distances = get_homolog_distances(
        human_embeddings[homologs.values[:, 0], :],
        yeast_embeddings[homologs.values[:, 1], :])
    # homolog_distances = np.sort(homolog_distances)[:400]
    homolog_distances = np.random.choice(homolog_distances, 400)

    # plt.hist(human_distances)
    # plt.show()

    print('Homolog distances')
    print('Mean: ', np.mean(homolog_distances))
    print('Std: ', np.std(homolog_distances))

    # plt.hist(homolog_distances)
    # plt.show()

