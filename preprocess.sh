#!/usr/bin/env bash

unzip raw_data.zip

##### SELECT LABELS TO BE USED FOR PREDICTION #####
python -m utils.select_labels --min_genes_per_class 200
cp data/mashup/raw/annotations/human/reduced_adjacency-22.txt cge/
python -m utils.select_labels --min_genes_per_class 30
cp data/mashup/raw/annotations/human/reduced_adjacency-211.txt cge/

##### PREPROCESS HOMOLOGS FILE #####
python -m utils.preprocess_homologs --data_dir ./data --output_dir ./data

##### GENERATE GRAPH INPUT FOR HoGEm #####
python utils.preprocess --output_dir cge/