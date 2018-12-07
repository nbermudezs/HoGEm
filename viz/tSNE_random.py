__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("embeddings_dir", None, "Path to the embeddings dir")
flags.DEFINE_integer("n_components", 2, "Number of components for tSNE")
flags.DEFINE_string("plot_filename", None, "Where to save the plot")
flags.DEFINE_string("homologs", "cge/ppi-homologs.txt",
                    "File containing the homolog adjacency")
flags.DEFINE_integer("n_points", 100,
                     "Points per organism to be plotted")


def main(_):
    print('Processing {}'.format(FLAGS.embeddings_dir))

    human_embs = np.load('{}/embs_{}-tSNE_2D.npy'.format(FLAGS.embeddings_dir, 'human'))
    yeast_embs = np.load('{}/embs_{}-tSNE_2D.npy'.format(FLAGS.embeddings_dir, 'yeast'))

    min_ = []
    max_ = []
    mean_ = []
    std_ = []

    human, yeast = None, None

    for k in range(100):
        human_indices = np.random.choice(range(human_embs.shape[0]), FLAGS.n_points)
        yeast_indices = np.random.choice(range(yeast_embs.shape[0]), FLAGS.n_points)

        human = human_embs[human_indices, :]
        yeast = yeast_embs[yeast_indices, :]
        dist = np.sqrt(np.sum((human - yeast) ** 2, axis=1))
        min_.append(np.min(dist))
        max_.append(np.max(dist))
        mean_.append(np.mean(dist))
        std_.append(np.std(dist))

    print('Min distance between pairs of genes:', np.mean(min_))
    print('Max distance between pairs of genes:', np.mean(max_))
    print('Average distance between pairs of genes:', np.mean(mean_))
    print('Std of distance between pairs of genes:', np.mean(std_))

    colors = np.random.rand(FLAGS.n_points)

    plt.scatter(human[:, 0], human[:, 1], marker='+', c=colors)
    plt.scatter(yeast[:, 0], yeast[:, 1], marker='x', c=colors)
    plt.legend(['human', 'yeast'])
    plt.title('Distance between random pairs of human and yeast gene embeddings')

    for idx in range(FLAGS.n_points):
        plt.plot([human[idx, 0], yeast[idx, 0]], [human[idx, 1], yeast[idx, 1]],
                 '--', c='grey')

    out = FLAGS.plot_filename or (FLAGS.embeddings_dir + '/random_plot.png')
    plt.savefig(out)
    # plt.show()
    print('=' * 80)


if __name__ == '__main__':
    tf.app.run()