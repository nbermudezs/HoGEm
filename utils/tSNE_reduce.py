__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("embeddings_dir", None, "Path to the embeddings dir")
flags.DEFINE_integer("n_components", 2, "Number of components for tSNE")
flags.DEFINE_list("organisms", ["human", "yeast"],
                  "List of organisms being processed")


def main(_):
    all_data = []
    sizes = {}
    for org in FLAGS.organisms:
        filepath = FLAGS.embeddings_dir + '/embs_{}.npy'.format(org)
        data = np.load(filepath, encoding='latin1')
        all_data.append(data)
        sizes[org] = data.shape[0]

    data = np.vstack(all_data)
    data = TSNE().fit_transform(data)

    offset = 0
    for org in FLAGS.organisms:
        org_embs = data[offset: offset + sizes[org], :]
        offset += sizes[org]
        np.save(FLAGS.embeddings_dir + '/embs_{}-tSNE_2D.npy'.format(org),
                org_embs)


if __name__ == '__main__':
    tf.app.run()
