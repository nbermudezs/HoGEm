__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import numpy as np
import pandas as pd
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "./data", "Root of input data")
flags.DEFINE_string("networks_dir", "./data/mashup/raw/networks",
                    "Path to the dir containing the network files")
flags.DEFINE_string("output_dir", "./data", "Output dir")


def convert_hsa_ensemble_ids_to_gene_id():
    filename = FLAGS.data_dir + '/human.ensembleId2symbol.txt'
    ensemble_id2symbol = pd.read_csv(filename,
                                     sep='\t', names=['id', 'symbol'])
    ensemble_id2symbol = {row['id']: row['symbol']
                          for _, row in ensemble_id2symbol.iterrows()}

    hsa_sc = pd.read_csv(FLAGS.data_dir + '/homologs/raw-human-yeast.txt',
                         sep='\t', names=['hsa', 'sc'])
    sc_hsa = pd.read_csv(FLAGS.data_dir + '/homologs/raw-yeast-human.txt',
                         sep='\t', names=['sc', 'hsa'])

    print('# of homologs: {}'.format(hsa_sc.shape[0]))

    for _, row in hsa_sc.iterrows():
        row['hsa'] = ensemble_id2symbol[row['hsa']]

    for _, row in sc_hsa.iterrows():
        row['hsa'] = ensemble_id2symbol[row['hsa']]

    hsa_sc.to_csv(FLAGS.output_dir + '/homologs/human-yeast.txt', sep='\t',
                  header=None, index=False)
    sc_hsa.to_csv(FLAGS.output_dir + '/homologs/yeast-human.txt', sep='\t',
                  header=None, index=False)


def filter_missing_nodes():
    hsa_sc = pd.read_csv(FLAGS.data_dir + '/homologs/human-yeast.txt',
                         sep='\t', names=['hsa', 'sc'])
    sc_hsa = pd.read_csv(FLAGS.data_dir + '/homologs/yeast-human.txt',
                         sep='\t', names=['sc', 'hsa'])

    # filter homologs not present in the human PPI network
    hsa_genes_ppi = pd.read_csv(
        FLAGS.networks_dir + '/human/human_string_genes.txt',
        names=['gene'])
    print('# of nodes in human PPI networks: {}'.format(hsa_genes_ppi.shape[0]))

    intersection = np.isin(hsa_sc.values[:, 0], hsa_genes_ppi.values)
    print('# of homologs present in human PPI network: {}'.format(
        np.count_nonzero(intersection)))
    missing = np.nonzero(~intersection)[0]
    hsa_sc = hsa_sc.drop(missing)
    sc_hsa = sc_hsa.drop(missing)

    # filter homologs not present in the yeast PPI network
    sc_genes_ppi = pd.read_csv(
        FLAGS.networks_dir + '/yeast/yeast_string_genes.txt',
        names=['gene'])
    print('# of nodes in yeast PPI networks: {}'.format(sc_genes_ppi.shape[0]))

    intersection = np.isin(hsa_sc.values[:, 1], sc_genes_ppi.values)
    print('# of homologs present in yeast PPI network: {}'.format(
        np.count_nonzero(intersection)))
    missing = np.nonzero(~intersection)[0]
    hsa_sc = hsa_sc.drop(missing)
    sc_hsa = sc_hsa.drop(missing)

    hsa_sc.to_csv(FLAGS.output_dir + '/homologs/human-yeast.txt', sep='\t',
                  header=None, index=False)
    sc_hsa.to_csv(FLAGS.output_dir + '/homologs/yeast-human.txt', sep='\t',
                  header=None, index=False)


def convert_to_indices():
    hsa_sc = pd.read_csv(FLAGS.data_dir + '/homologs/human-yeast.txt',
                         sep='\t', names=['hsa', 'sc'])
    sc_hsa = pd.read_csv(FLAGS.data_dir + '/homologs/yeast-human.txt',
                         sep='\t', names=['sc', 'hsa'])

    hsa_genes_ppi = pd.read_csv(
        FLAGS.networks_dir + '/human/human_string_genes.txt',
        names=['gene']).values

    sc_genes_ppi = pd.read_csv(
        FLAGS.networks_dir + '/yeast/yeast_string_genes.txt',
        names=['gene']).values

    for _, row in hsa_sc.iterrows():
        row['hsa'] = np.argmax(hsa_genes_ppi == row['hsa'])
        row['sc'] = np.argmax(sc_genes_ppi == row['sc'])

    for _, row in sc_hsa.iterrows():
        row['hsa'] = np.argmax(hsa_genes_ppi == row['hsa'])
        row['sc'] = np.argmax(sc_genes_ppi == row['sc'])

    hsa_sc.to_csv(FLAGS.output_dir + '/homologs/human-yeast.txt', sep='\t',
                  header=None, index=False)
    sc_hsa.to_csv(FLAGS.output_dir + '/homologs/yeast-human.txt', sep='\t',
                  header=None, index=False)


def main(_):
    convert_hsa_ensemble_ids_to_gene_id()
    filter_missing_nodes()
    convert_to_indices()


if __name__ == '__main__':
    tf.app.run()
