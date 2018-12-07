from __future__ import print_function

__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import json
import networkx as nx
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from networkx.readwrite import json_graph
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.joblib import dump


assert(nx.__version__ <= (1, 11))


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("networks_dir", "data/mashup/raw/networks/",
                    "Directory containing the different network adjacencies")
flags.DEFINE_string("annotations_dir", "data/mashup/raw/annotations/",
                    "Directory containing the files to extract labels")
flags.DEFINE_string("homologs_dir", "data/",
                    "Directory containing the homolog file(s)")
flags.DEFINE_string("output_dir", "cge/",
                    "Directory where the processed files will be stored")


WALK_LEN = 5
N_WALKS = 5


def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs


def get_all_genes(org):
    return pd.read_csv(FLAGS.networks_dir + org + '/' + org + '_string_genes.txt', header=None, names=['gene'])


def read_adjacency_file(filepath):
    data = pd.read_csv(filepath, sep='\t', header=None, names=['u', 'v', 'w'], dtype={'u': int, 'v': int, 'w': float})
    data['u'] = data['u'] - 1
    data['v'] = data['v'] - 1

    return data


def gene_class_map(gene, annotations, n_classes):
    indices = annotations[annotations['gene'] == gene]['GO'].values
    class_map = np.zeros(n_classes, dtype=int)
    class_map[indices] = 1
    return class_map.tolist()


if __name__ == '__main__':
    import os
    G = nx.Graph()
    id_map = {}
    class_map = {}
    OFFSET = 1000000

    homolog_root = FLAGS.homologs_dir
    homologs_filepath = homolog_root + '/human-yeast.txt'
    data = pd.read_csv(homologs_filepath, sep='\t', header=None,
                       names=['human', 'yeast'])
    data['yeast'] = data['yeast'] + OFFSET
    data.to_csv(homolog_root + '/human-yeast_corrected.txt', sep='\t', header=None, index=False)
    human_annotations_path = FLAGS.annotations_dir + '/human/reduced_adjacency-22.txt'
    human_annotations = pd.read_csv(human_annotations_path, sep='\t', header=None, names=['gene', 'GO'])

    yeast_annotations_path = FLAGS.annotations_dir + '/yeast/yeast_mips_level1_adjacency.txt'
    yeast_annotations = pd.read_csv(yeast_annotations_path, sep='\t', header=None, names=['gene', 'GO'])
    yeast_annotations = yeast_annotations + OFFSET

    human_annotations['gene'] = human_annotations['gene'] - 1
    yeast_annotations['gene'] = yeast_annotations['gene'] - 1
    labels = pd.concat([human_annotations, yeast_annotations])
    label_encoder = LabelEncoder()
    labels['GO'] = label_encoder.fit_transform(labels['GO'])
    n_classes = label_encoder.classes_.size

    human_genes = get_all_genes('human')
    n_human_genes = human_genes.shape[0]

    yeast_genes = get_all_genes('yeast')
    n_yeast_genes = yeast_genes.shape[0]

    n_networks = 6

    network_id = 0
    node_id = 0
    node_id_set = set()
    for file in os.listdir(FLAGS.networks_dir + 'human'):
        if not file.endswith('adjacency.txt'):
            continue
        for i, gene in enumerate(range(n_human_genes)):
            node_id = '{}_{}'.format(network_id, gene)
            node_id_set.add(node_id)
            val = 'cooccurence' in file
            G.add_node(node_id, test=False, val=val)
            class_map[node_id] = gene_class_map(gene, labels, n_classes)
            id_map[node_id] = network_id * n_human_genes + i
        network_id += 1

    id_offset = n_networks * n_human_genes

    network_id = 0
    for file in os.listdir(FLAGS.networks_dir + 'yeast'):
        if not file.endswith('adjacency.txt'):
            continue
        for i, gene in enumerate(range(n_yeast_genes)):
            node_id = '{}_{}'.format(n_networks + network_id, gene + OFFSET)
            node_id_set.add(node_id)
            val = 'cooccurence' in file
            G.add_node(node_id, test=False, val=val)
            class_map[node_id] = gene_class_map(gene, labels, n_classes)
            id_map[node_id] = id_offset + network_id * n_yeast_genes + i
        network_id += 1

    network_id = 0

    for file in os.listdir(FLAGS.networks_dir + 'human'):
        if not file.endswith('adjacency.txt'):
            continue
        print('Processing {} graph'.format(file))
        full_data = read_adjacency_file(
            FLAGS.networks_dir + 'human/' + file)
        for data in np.array_split(full_data, full_data.shape[0] / 20000 + 1):
            for i, (u, v, w) in enumerate(data.values):
                u = int(u)
                v = int(v)
                print('Processing link {0:6d} out of {1:6d}'.format(i, data.values.shape[0]), end='\r')
                u_id = '{}_{}'.format(network_id, u)
                v_id = '{}_{}'.format(network_id, v)
                G.add_edge(u_id, v_id, weight=w)
        network_id += 1

    network_id = 0
    for file in os.listdir(FLAGS.networks_dir + 'yeast'):
        if not file.endswith('adjacency.txt'):
            continue
        print('Processing {} graph'.format(file))
        full_data = read_adjacency_file(
            FLAGS.networks_dir + 'yeast/' + file)
        for data in np.array_split(full_data, full_data.shape[0] / 20000 + 1):
            for i, (u, v, w) in enumerate(data.values):
                u = int(u) + OFFSET
                v = int(v) + OFFSET
                print('Processing link {0:6d} out of {1:6d}'.format(i, data.values.shape[0]), end='\r')
                u_id = '{}_{}'.format(n_networks + network_id, u)
                v_id = '{}_{}'.format(n_networks + network_id, v)
                G.add_edge(u_id, v_id, weight=w)
        network_id += 1

    print('=' * 120)
    print('Saving files...')
    walks = run_random_walks(G, G.nodes())

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    with open(FLAGS.output_dir + '/ppi-G.json', 'w') as f:
        json.dump(json_graph.node_link_data(G), f)

    with open(FLAGS.output_dir + '/ppi-id_map.json', 'w') as f:
        json.dump(id_map, f)

    with open(FLAGS.output_dir + '/ppi-class_map.json', 'w') as f:
        json.dump(class_map, f)

    with open(FLAGS.output_dir + '/ppi-walks.txt', 'w') as f:
        for walk in walks:
            f.write('\t'.join(walk) + '\n')

    dump(label_encoder, FLAGS.output_dir + '/ppi-label_encoder.pkl')
    print('DONE')
