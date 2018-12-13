__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import numpy as np
import pandas as pd


N_HUMAN_GENES = 18362
N_YEAST_GENES = 6400
OFFSET = 1000000


def generate_pseudo_homologs(n_homologs=933):
    used = set()
    result = []

    human_genes = range(N_HUMAN_GENES)
    yeast_genes = range(N_YEAST_GENES)
    while len(result) < n_homologs:
        g1, g2 = np.random.choice(human_genes), np.random.choice(yeast_genes)
        if (g1, g2) not in used:
            used.add((g1, g2))
            result.append((g1, OFFSET + g2))

    return result


if __name__ == '__main__':
    pairs = generate_pseudo_homologs()
    df = pd.DataFrame.from_records(pairs, columns=['hsa', 'cs'])
    df.to_csv('ppi-homologs.txt', sep='\t', index=False, header=None)