## HoGEm: Homology-guided Gene Embedding

#### Authors: [Nestor A. Bermudez](https://nbermudezs.com/cv) (nab6@illinois.edu)

### Overview

This work is an extension of the work developed by Hamilton, Ying and Leskovec at Stanford (GraphSAGE).
As a result, it supports all command line and docker configurations present in GraphSAGE so please refer to [GraphSAGE](https://github.com/williamleif/GraphSAGE/tree/master/graphsage) for
detailed instructions.

### Requirements

Recent versions of TensorFlow, numpy, scipy, sklearn, and networkx are required (but networkx must be <=1.11). You can install all the required packages using the following command:

	$ pip install -r requirements.txt

To guarantee that you have the right package versions, you can use [docker](https://docs.docker.com/) to easily set up a virtual environment. See the Docker subsection below for more info.

### Running the code

The `paper_experiments.sh` bash script contains all the commands needed to reproduce the results
described in our paper.

#### The data
If you would like to preprocess the raw data on your own feel free to use the *raw_data.zip* in this repo. 
All the steps needed to process the raw data can be seen in *preprocess.sh*.

On the other hand, if you just want to use the data used for our experiments,
just download the processed files from [this Box folder](https://uofi.box.com/s/c0t6c16ld1uki4khymv8drzw68thitvd).

#### Input format
As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

* <train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* <train_prefix>-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* <train_prefix>-walks.txt [optional] --- A text file specifying random walk co-occurrences (one pair per line) (*only for unsupervised version of graphsage)
* <train_prefix>-homologs.txt -- A text file containing mapping between genes from species 1 to genes in species 2. Species 2 gene ids are offseted by 1000000 to avoid
collisions with the gene ids of species 1.

#### Model variants

Although GraphSAGE supports multiple model variants, as described below, only the maxpool variant is used in our paper
because it is the one that performed the best under the original GraphSAGE setup.

The user must also specify a --model, the variants of which are described in detail in the paper:
* graphsage_mean -- GraphSage with mean-based aggregator
* graphsage_seq -- GraphSage with LSTM-based aggregator
* graphsage_maxpool -- GraphSage with max-pooling aggregator (as described in the NIPS 2017 paper)
* graphsage_meanpool -- GraphSage with mean-pooling aggregator (a variant of the pooling aggregator, where the element-wie mean replaces the element-wise max).
* gcn -- GraphSage with GCN-based aggregator
* n2v -- an implementation of [DeepWalk](https://arxiv.org/abs/1403.6652) (called n2v for short in the code.)

#### Logging directory
Finally, a --base_log_dir should be specified (it defaults to the current directory).
The output of the model and log files will be stored in a subdirectory of the base_log_dir.
The path to the logged data will be of the form `<sup/unsup>-<data_prefix>/graphsage-<model_description>/`.
The supervised model will output F1 scores, while the unsupervised model will train embeddings and store them.
The unsupervised embeddings will be stored in a numpy formated file named val.npy with val.txt specifying the order of embeddings as a per-line list of node ids.
Note that the full log outputs and stored embeddings can be 5-10Gb in size (on the full data when running with the unsupervised variant).

#### Acknowledgements

The original version of this code base was originally forked from https://github.com/williamleif/GraphSAGE/tree/master/graphsage, thanks to Hamilton et al. to make their code available and allow their work to be extended.
