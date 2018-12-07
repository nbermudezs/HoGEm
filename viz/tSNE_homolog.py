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
flags.DEFINE_list("organisms", ["human", "yeast"],
                  "Organisms being considered")
flags.DEFINE_string("homologs", "cge/ppi-homologs.txt",
                    "File containing the homolog adjacency")
flags.DEFINE_integer("n_points", 100,
                     "Points per organism to be plotted")


def main(_):
    print('Processing {}'.format(FLAGS.embeddings_dir))
    all_data = []
    all_labels = []

    homologs = pd.read_csv(FLAGS.homologs, sep='\t', header=None,
                           names=['human', 'yeast'])
    homologs['yeast'] -= 1000000
    homologs = homologs.values

    # indices = np.random.choice(range(homologs.shape[0]), FLAGS.n_points)
    indices = range(homologs.shape[0])

    for i, org in enumerate(FLAGS.organisms):
        filepath = '{}/embs_{}-tSNE_2D.npy'.format(FLAGS.embeddings_dir, org)
        if not os.path.exists(filepath):
            filepath = '{}/embs_{}.npy'.format(FLAGS.embeddings_dir, org)

        data = np.load(filepath, encoding='latin1')[homologs[indices, i], :]
        labels = range(FLAGS.n_points)
        all_data.append(data)
        all_labels.append(labels)

    data = np.vstack(all_data)

    if data.shape[1] > FLAGS.n_components:
        data = TSNE(n_components=FLAGS.n_components).fit_transform(data)
    human, yeast = data[:len(indices), :], data[len(indices):, :]
    dist = np.sqrt(np.sum((human - yeast) ** 2, axis=1))

    print('Min distance between homolog genes:', np.min(dist))
    print('Max distance between homolog genes:', np.max(dist))
    print('Average distance between homolog genes:', np.mean(dist))
    print('Std of distance between homolog genes:', np.std(dist))

    indices = np.random.choice(range(len(indices)), FLAGS.n_points)
    colors = np.random.rand(FLAGS.n_points)

    plt.scatter(human[indices, 0], human[indices, 1], marker='+', c=colors)
    plt.scatter(yeast[indices, 0], yeast[indices, 1], marker='x', c=colors)
    plt.legend(FLAGS.organisms)
    plt.title('Distance between human and yeast homologs')

    for i in range(len(indices)):
        idx = indices[i]
        plt.plot([human[idx, 0], yeast[idx, 0]], [human[idx, 1], yeast[idx, 1]], '--', c='grey')

    out = FLAGS.plot_filename or (FLAGS.embeddings_dir + '/homologs_plot.png')
    plt.savefig(out)
    # plt.show()
    print('=' * 80)


if __name__ == '__main__':
    tf.app.run()