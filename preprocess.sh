#!/usr/bin/env bash

unzip raw_data.zip

##### SELECT LABELS TO BE USED FOR PREDICTION #####
python -m utils.select_labels --min_genes_per_class 200
python -m utils.select_labels --min_genes_per_class 30

##### PREPROCESS HOMOLOGS FILE #####
python -m utils.preprocess_homologs --data_dir ./data --output_dir ./data

##### GENERATE GRAPH INPUT FOR HoGEm #####
python utils.preprocess --output_dir cge/