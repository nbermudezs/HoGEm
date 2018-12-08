__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("embeddings_dir", None, "Path to the embeddings dir")


def main(_):
    print('Processing {}...'.format(FLAGS.embeddings_dir))

    human_embs = np.load('{}/embs_{}-tSNE_2D.npy'.format(FLAGS.embeddings_dir, 'human'))
    yeast_embs = np.load('{}/embs_{}-tSNE_2D.npy'.format(FLAGS.embeddings_dir, 'yeast'))

    plt.scatter(human_embs[:, 0], human_embs[:, 1], marker='+', alpha=0.5)
    plt.scatter(yeast_embs[:, 0], yeast_embs[:, 1], marker='x', alpha=0.5)
    plt.legend(['human', 'yeast'])
    plt.title('All human and yeast genes')
    plt.savefig(FLAGS.embeddings_dir + '/all_genes.png')
    plt.cla()

    plt.scatter(human_embs[:, 0], human_embs[:, 1], marker='+', alpha=0.5)
    plt.title('All human genes')
    plt.savefig(FLAGS.embeddings_dir + '/all_human_genes.png')
    plt.cla()

    plt.scatter(yeast_embs[:, 0], yeast_embs[:, 1], marker='x', alpha=0.5)
    plt.title('All yeast genes')
    plt.savefig(FLAGS.embeddings_dir + '/all_yeast_genes.png')



if __name__ == '__main__':
    tf.app.run()