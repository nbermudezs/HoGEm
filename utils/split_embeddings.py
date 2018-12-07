__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("embeddings_path", None,
                    "Path to the embeddings file")
flags.DEFINE_string("output_dir", None,
                    "Path to the output dir")
flags.DEFINE_integer("n_graphs", 6,
                    "Number of graphs present per organism")
flags.DEFINE_list("organisms", ["human", "yeast"],
                  "List of organisms present in the data")
flags.DEFINE_list("org_nodes", [18362, 6400],
                  "Number of nodes per organism network")


def main(_):
    all_embs = np.load(FLAGS.embeddings_path)
    n_dims = all_embs.shape[1]
    print("Shape of embeddings: ", all_embs.shape)
    assert (all_embs.shape[0] == FLAGS.n_graphs * np.sum(FLAGS.org_nodes))

    offset = 0
    for org, size in zip(FLAGS.organisms, FLAGS.org_nodes):
        org_embs = all_embs[offset:offset + FLAGS.n_graphs * size]
        org_embs = org_embs.reshape(FLAGS.n_graphs, size, -1)
        org_embs = np.mean(org_embs, axis=0)
        assert (org_embs.shape == (size, n_dims))

        org_embs.dump('{}/embs_{}.npy'.format(FLAGS.output_dir, org))
        offset += FLAGS.n_graphs * size


if __name__ == '__main__':
    tf.app.run()