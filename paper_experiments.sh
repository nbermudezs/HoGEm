#!/usr/bin/env bash

if [ ! -d mse-homolog-loss ]; then
mkdir mse-homolog-loss
fi

# MSE homolog loss
if [ ! -f mse-homolog-loss/mse_b512_e20/val.npy ]; then
echo 'MSE homolog loss with batch_size 512'
python -m graphsage.unsupervised_train --train_prefix ./cge/ppi --homolog_loss mse \
    --model graphsage_maxpool --max_total_steps 100000000000 --epochs 10 --validate_iter 10 \
    --identity_dim 128 --dim_1 400 --dim_2 400 --dropout 0.05
mkdir mse-homolog-loss/mse_b512_e20
mv unsup-cge/graphsage_maxpool_small_0.000010/val.npy mse-homolog-loss/mse_b512_e20/val.npy
rm -rf unsup-cge
echo 'Embeddings have been saved to mse-homolog-loss/mse_b512_e20'
fi

if [ ! -f mse-homolog-loss/mse_b256_e20/val.npy ]; then
echo 'MSE homolog loss with batch_size 256'
python -m graphsage.unsupervised_train --train_prefix ./cge/ppi --homolog_loss mse \
    --model graphsage_maxpool --batch_size 256 --max_total_steps 100000000000 --epochs 10 \
    --validate_iter 10 --identity_dim 128 --dim_1 400 --dim_2 400 --dropout 0.05
mkdir mse-homolog-loss/mse_b256_e20
mv unsup-cge/graphsage_maxpool_small_0.000010/val.npy mse-homolog-loss/mse_b256_e20/val.npy
rm -rf unsup-cge
echo 'Embeddings have been saved to mse-homolog-loss/mse_b256_e20'
fi


if [ ! -d cross-homolog-loss ]; then
mkdir cross-homolog-loss
fi

# dot product homolog loss
if [ ! -f cross-homolog-loss/cross_b512_e20/val.npy ]; then
echo 'cross product homolog loss with batch_size 512'
python -m graphsage.unsupervised_train --train_prefix ./cge/ppi --homolog_loss cross \
    --model graphsage_maxpool --max_total_steps 100000000000 --epochs 10 --validate_iter 10 \
    --identity_dim 128 --dim_1 400 --dim_2 400 --dropout 0.05
mkdir cross-homolog-loss/cross_b512_e20
mv unsup-cge/graphsage_maxpool_small_0.000010/val.npy cross-homolog-loss/cross_b512_e20/val.npy
rm -rf unsup-cge
echo 'Embeddings have been saved to cross-homolog-loss/cross_b512_e20'
fi


# dot product homolog loss
if [ ! -f cross-homolog-loss/cross_b256_e20/val.npy ]; then
echo 'cross product homolog loss with batch_size 256'
python -m graphsage.unsupervised_train --train_prefix ./cge/ppi --homolog_loss cross \
    --model graphsage_maxpool --batch_size 256 --max_total_steps 100000000000 --epochs 10 \
    --validate_iter 10 --identity_dim 128 --dim_1 400 --dim_2 400 --dropout 0.05
mkdir cross-homolog-loss/cross_b256_e20
mv unsup-cge/graphsage_maxpool_small_0.000010/val.npy cross-homolog-loss/cross_b256_e20/val.npy
rm -rf unsup-cge
echo 'Embeddings have been saved to cross-homolog-loss/cross_b256_e20'
fi

if [ ! -d dot-homolog-loss ]; then
mkdir dot-homolog-loss
fi

# dot product homolog loss
if [ ! -f dot-homolog-loss/dot_b512_e20/val.npy ]; then
echo 'dot product homolog loss with batch_size 512'
python -m graphsage.unsupervised_train --train_prefix ./cge/ppi --homolog_loss dot \
    --model graphsage_maxpool --max_total_steps 100000000000 --epochs 10 --validate_iter 10 \
    --identity_dim 128 --dim_1 400 --dim_2 400 --dropout 0.05
mkdir dot-homolog-loss/dot_b512_e20
mv unsup-cge/graphsage_maxpool_small_0.000010/val.npy dot-homolog-loss/dot_b512_e20/val.npy
rm -rf unsup-cge
echo 'Embeddings have been saved to dot-homolog-loss/dot_b512_e20'
fi


# dot product homolog loss
if [ ! -f dot-homolog-loss/dot_b256_e20/val.npy ]; then
echo 'dot product homolog loss with batch_size 256'
python -m graphsage.unsupervised_train --train_prefix ./cge/ppi --homolog_loss dot \
    --model graphsage_maxpool --batch_size 256 --max_total_steps 100000000000 --epochs 10 \
    --validate_iter 10 --identity_dim 128 --dim_1 400 --dim_2 400 --dropout 0.05
mkdir dot-homolog-loss/dot_b256_e20
mv unsup-cge/graphsage_maxpool_small_0.000010/val.npy dot-homolog-loss/dot_b256_e20/val.npy
rm -rf unsup-cge
echo 'Embeddings have been saved to dot-homolog-loss/dot_b256_e20'
fi


# dot product homolog loss
if [ ! -f dot-homolog-loss/dot_b512_e20_hi1000/val.npy ]; then
echo 'dot product homolog loss with homolog importance set to 1000'
python -m graphsage.unsupervised_train --train_prefix ./cge/ppi --homolog_loss dot \
    --model graphsage_maxpool --batch_size 512 --max_total_steps 100000000000 --epochs 10 \
    --validate_iter 10 --identity_dim 128 --dim_1 400 --dim_2 400 --dropout 0.05 \
    --homolog_importance 1000
mkdir dot-homolog-loss/dot_b512_e20_hi1000
mv unsup-cge/graphsage_maxpool_small_0.000010/val.npy dot-homolog-loss/dot_b512_e20_hi1000/val.npy
rm -rf unsup-cge
echo 'Embeddings have been saved to dot-homolog-loss/dot_b512_e20_hi1000'
fi


if [ ! -f dot-homolog-loss/dot_b256_e20_hi1000/val.npy ]; then
echo 'dot product homolog loss with homolog importance set to 1000'
python -m graphsage.unsupervised_train --train_prefix ./cge/ppi --homolog_loss dot \
    --model graphsage_maxpool --batch_size 256 --max_total_steps 100000000000 --epochs 10 \
    --validate_iter 10 --identity_dim 128 --dim_1 400 --dim_2 400 --dropout 0.05 \
    --homolog_importance 1000
mkdir dot-homolog-loss/dot_b256_e20_hi1000
mv unsup-cge/graphsage_maxpool_small_0.000010/val.npy dot-homolog-loss/dot_b256_e20_hi1000/val.npy
rm -rf unsup-cge
echo 'Embeddings have been saved to dot-homolog-loss/dot_b256_e20_hi1000'
fi


##### SPLIT EMBEDDINGS PER ORGANISM #####
if [ ! -f mse-homolog-loss/mse_b512_e20/embs_human.npy ]; then
python -m utils.split_embeddings --embeddings_path mse-homolog-loss/mse_b512_e20/val.npy \
    --output_dir mse-homolog-loss/mse_b512_e20
fi
if [ ! -f mse-homolog-loss/mse_b256_e20/embs_human.npy ]; then
python -m utils.split_embeddings --embeddings_path mse-homolog-loss/mse_b256_e20/val.npy \
    --output_dir mse-homolog-loss/mse_b256_e20
fi
if [ ! -f cross-homolog-loss/cross_b512_e20/embs_human.npy ]; then
python -m utils.split_embeddings --embeddings_path cross-homolog-loss/cross_b512_e20/val.npy \
    --output_dir cross-homolog-loss/cross_b512_e20
fi
if [ ! -f cross-homolog-loss/cross_b256_e20/embs_human.npy ]; then
python -m utils.split_embeddings --embeddings_path cross-homolog-loss/cross_b256_e20/val.npy \
    --output_dir cross-homolog-loss/cross_b256_e20
fi
if [ ! -f dot-homolog-loss/dot_b512_e20/embs_human.npy ]; then
python -m utils.split_embeddings --embeddings_path dot-homolog-loss/dot_b512_e20/val.npy \
    --output_dir dot-homolog-loss/dot_b512_e20
fi
if [ ! -f dot-homolog-loss/dot_b256_e20/embs_human.npy ]; then
python -m utils.split_embeddings --embeddings_path dot-homolog-loss/dot_b256_e20/val.npy \
    --output_dir dot-homolog-loss/dot_b256_e20
fi
if [ ! -f dot-homolog-loss/dot_b512_e20_hi1000/embs_human.npy ]; then
python -m utils.split_embeddings --embeddings_path dot-homolog-loss/dot_b512_e20_hi1000/val.npy \
    --output_dir dot-homolog-loss/dot_b512_e20_hi1000
fi
if [ ! -f dot-homolog-loss/dot_b256_e20_hi1000/embs_human.npy ]; then
python -m utils.split_embeddings --embeddings_path dot-homolog-loss/dot_b256_e20_hi1000/val.npy \
    --output_dir dot-homolog-loss/dot_b256_e20_hi1000
fi


##### COMPUTE 2D tSNE EMBEDDINGS #####
if [ ! -f mse-homolog-loss/mse_b512_e20/embs_human-tSNE_2D.npy ]; then
python -m utils.tSNE_reduce --embeddings_dir mse-homolog-loss/mse_b512_e20
fi
if [ ! -f mse-homolog-loss/mse_b256_e20/embs_human-tSNE_2D.npy ]; then
python -m utils.tSNE_reduce --embeddings_dir mse-homolog-loss/mse_b256_e20
fi
if [ ! -f cross-homolog-loss/cross_b512_e20/embs_human-tSNE_2D.npy ]; then
python -m utils.tSNE_reduce --embeddings_dir cross-homolog-loss/cross_b512_e20
fi
if [ ! -f cross-homolog-loss/cross_b256_e20/embs_human-tSNE_2D.npy ]; then
python -m utils.tSNE_reduce --embeddings_dir cross-homolog-loss/cross_b256_e20
fi
if [ ! -f dot-homolog-loss/dot_b512_e20/embs_human-tSNE_2D.npy ]; then
python -m utils.tSNE_reduce --embeddings_dir dot-homolog-loss/dot_b512_e20
fi
if [ ! -f dot-homolog-loss/dot_b256_e20/embs_human-tSNE_2D.npy ]; then
python -m utils.tSNE_reduce --embeddings_dir dot-homolog-loss/dot_b256_e20
fi
if [ ! -f dot-homolog-loss/dot_b512_e20_hi1000/embs_human-tSNE_2D.npy ]; then
python -m utils.tSNE_reduce --embeddings_dir dot-homolog-loss/dot_b512_e20_hi1000
fi
if [ ! -f dot-homolog-loss/dot_b256_e20_hi1000/embs_human-tSNE_2D.npy ]; then
python -m utils.tSNE_reduce --embeddings_dir dot-homolog-loss/dot_b256_e20_hi1000
fi


##### CREATING t-SNE VISUALIZATIONS #####
# homologs
if [ -f summary.txt ]; then
    rm summary.txt
fi
python -m viz.tSNE_homolog --embeddings_dir mse-homolog-loss/mse_b512_e20 >> summary.txt
python -m viz.tSNE_homolog --embeddings_dir mse-homolog-loss/mse_b256_e20 >> summary.txt
python -m viz.tSNE_homolog --embeddings_dir cross-homolog-loss/cross_b512_e20 >> summary.txt
python -m viz.tSNE_homolog --embeddings_dir cross-homolog-loss/cross_b256_e20 >> summary.txt
python -m viz.tSNE_homolog --embeddings_dir dot-homolog-loss/dot_b512_e20 >> summary.txt
python -m viz.tSNE_homolog --embeddings_dir dot-homolog-loss/dot_b256_e20 >> summary.txt
python -m viz.tSNE_homolog --embeddings_dir dot-homolog-loss/dot_b512_e20_hi1000 >> summary.txt
python -m viz.tSNE_homolog --embeddings_dir dot-homolog-loss/dot_b256_e20_hi1000 >> summary.txt

# random pairs as reference
echo '================= RANDOM =================' >> summary.txt
python -m viz.tSNE_random --embeddings_dir mse-homolog-loss/mse_b512_e20 >> summary.txt
python -m viz.tSNE_random --embeddings_dir mse-homolog-loss/mse_b256_e20 >> summary.txt
python -m viz.tSNE_random --embeddings_dir cross-homolog-loss/cross_b512_e20 >> summary.txt
python -m viz.tSNE_random --embeddings_dir cross-homolog-loss/cross_b256_e20 >> summary.txt
python -m viz.tSNE_random --embeddings_dir dot-homolog-loss/dot_b512_e20 >> summary.txt
python -m viz.tSNE_random --embeddings_dir dot-homolog-loss/dot_b256_e20 >> summary.txt
python -m viz.tSNE_random --embeddings_dir dot-homolog-loss/dot_b512_e20_hi1000 >> summary.txt
python -m viz.tSNE_random --embeddings_dir dot-homolog-loss/dot_b256_e20_hi1000 >> summary.txt


python -m viz.tSNE_all --embeddings_dir mse-homolog-loss/mse_b512_e20
python -m viz.tSNE_all --embeddings_dir mse-homolog-loss/mse_b256_e20
python -m viz.tSNE_all --embeddings_dir cross-homolog-loss/cross_b512_e20
python -m viz.tSNE_all --embeddings_dir cross-homolog-loss/cross_b256_e20
python -m viz.tSNE_all --embeddings_dir dot-homolog-loss/dot_b512_e20
python -m viz.tSNE_all --embeddings_dir dot-homolog-loss/dot_b256_e20
python -m viz.tSNE_all --embeddings_dir dot-homolog-loss/dot_b512_e20_hi1000
python -m viz.tSNE_all --embeddings_dir dot-homolog-loss/dot_b256_e20_hi1000


##### PREPARE FOR CLASSIFIER TRAINING #####
if [ ! -f mse-homolog-loss/mse_b512_e20/c22/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-22.txt \
    --embeddings_filepath mse-homolog-loss/mse_b512_e20/embs_human.npy \
    --output_dir mse-homolog-loss/mse_b512_e20/c22
fi

if [ ! -f mse-homolog-loss/mse_b512_e20/c211/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-211.txt \
    --embeddings_filepath mse-homolog-loss/mse_b512_e20/embs_human.npy \
    --output_dir mse-homolog-loss/mse_b512_e20/c211
fi


if [ ! -f mse-homolog-loss/mse_b256_e20/c22/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-22.txt \
    --embeddings_filepath mse-homolog-loss/mse_b256_e20/embs_human.npy \
    --output_dir mse-homolog-loss/mse_b256_e20/c22
fi

if [ ! -f mse-homolog-loss/mse_b256_e20/c211/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-211.txt \
    --embeddings_filepath mse-homolog-loss/mse_b256_e20/embs_human.npy \
    --output_dir mse-homolog-loss/mse_b256_e20/c211
fi



if [ ! -f cross-homolog-loss/cross_b512_e20/c22/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-22.txt \
    --embeddings_filepath cross-homolog-loss/cross_b512_e20/embs_human.npy \
    --output_dir cross-homolog-loss/cross_b512_e20/c22
fi

if [ ! -f cross-homolog-loss/cross_b512_e20/c211/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-211.txt \
    --embeddings_filepath cross-homolog-loss/cross_b512_e20/embs_human.npy \
    --output_dir cross-homolog-loss/cross_b512_e20/c211
fi




if [ ! -f cross-homolog-loss/cross_b256_e20/c22/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-22.txt \
    --embeddings_filepath cross-homolog-loss/cross_b256_e20/embs_human.npy \
    --output_dir cross-homolog-loss/cross_b256_e20/c22
fi

if [ ! -f cross-homolog-loss/cross_b256_e20/c211/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-211.txt \
    --embeddings_filepath cross-homolog-loss/cross_b256_e20/embs_human.npy \
    --output_dir cross-homolog-loss/cross_b256_e20/c211
fi




if [ ! -f dot-homolog-loss/dot_b512_e20/c22/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-22.txt \
    --embeddings_filepath dot-homolog-loss/dot_b512_e20/embs_human.npy \
    --output_dir dot-homolog-loss/dot_b512_e20/c22
fi

if [ ! -f dot-homolog-loss/dot_b512_e20/c211/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-211.txt \
    --embeddings_filepath dot-homolog-loss/dot_b512_e20/embs_human.npy \
    --output_dir dot-homolog-loss/dot_b512_e20/c211
fi




if [ ! -f dot-homolog-loss/dot_b256_e20/c22/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-22.txt \
    --embeddings_filepath dot-homolog-loss/dot_b256_e20/embs_human.npy \
    --output_dir dot-homolog-loss/dot_b256_e20/c22
fi

if [ ! -f dot-homolog-loss/dot_b256_e20/c211/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-211.txt \
    --embeddings_filepath dot-homolog-loss/dot_b256_e20/embs_human.npy \
    --output_dir dot-homolog-loss/dot_b256_e20/c211
fi




if [ ! -f dot-homolog-loss/dot_b512_e20_hi1000/c22/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-22.txt \
    --embeddings_filepath dot-homolog-loss/dot_b512_e20_hi1000/embs_human.npy \
    --output_dir dot-homolog-loss/dot_b512_e20_hi1000/c22
fi

if [ ! -f dot-homolog-loss/dot_b512_e20_hi1000/c211/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-211.txt \
    --embeddings_filepath dot-homolog-loss/dot_b512_e20_hi1000/embs_human.npy \
    --output_dir dot-homolog-loss/dot_b512_e20_hi1000/c211
fi




if [ ! -f dot-homolog-loss/dot_b256_e20_hi1000/c22/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-22.txt \
    --embeddings_filepath dot-homolog-loss/dot_b256_e20_hi1000/embs_human.npy \
    --output_dir dot-homolog-loss/dot_b256_e20_hi1000/c22
fi

if [ ! -f dot-homolog-loss/dot_b256_e20_hi1000/c211/human-features.npy ]; then
python -m utils.prepare_for_classifier --organism human \
    --labels_filepath cge/reduced_adjacency-211.txt \
    --embeddings_filepath dot-homolog-loss/dot_b256_e20_hi1000/embs_human.npy \
    --output_dir dot-homolog-loss/dot_b256_e20_hi1000/c211
fi

##### TRAINING CLASSIFIERS #####
if [ ! -f mse-homolog-loss/mse_b512_e20/c22/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir mse-homolog-loss/mse_b512_e20/c22
fi
if [ ! -f mse-homolog-loss/mse_b512_e20/c211/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir mse-homolog-loss/mse_b512_e20/c211
fi

if [ ! -f dot-homolog-loss/dot_b512_e20_hi1000/c22/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir mse-homolog-loss/mse_b256_e20/c22
fi
if [ ! -f mse-homolog-loss/mse_b256_e20/c211/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir mse-homolog-loss/mse_b256_e20/c211
fi

if [ ! -f cross-homolog-loss/cross_b512_e20/c22/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir cross-homolog-loss/cross_b512_e20/c22
fi
if [ ! -f cross-homolog-loss/cross_b512_e20/c211/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir cross-homolog-loss/cross_b512_e20/c211
fi

if [ ! -f cross-homolog-loss/cross_b256_e20/c22/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir cross-homolog-loss/cross_b256_e20/c22
fi
if [ ! -f cross-homolog-loss/cross_b256_e20/c211/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir cross-homolog-loss/cross_b256_e20/c211
fi

if [ ! -f dot-homolog-loss/dot_b512_e20/c22/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir dot-homolog-loss/dot_b512_e20/c22
fi
if [ ! -f dot-homolog-loss/dot_b512_e20/c211/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir dot-homolog-loss/dot_b512_e20/c211
fi

if [ ! -f dot-homolog-loss/dot_b256_e20/c22/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir dot-homolog-loss/dot_b256_e20/c22
fi
if [ ! -f dot-homolog-loss/dot_b256_e20/c211/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir dot-homolog-loss/dot_b256_e20/c211
fi

if [ ! -f dot-homolog-loss/dot_b512_e20_hi1000/c22/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir dot-homolog-loss/dot_b512_e20_hi1000/c22
fi
if [ ! -f dot-homolog-loss/dot_b512_e20_hi1000/c211/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir dot-homolog-loss/dot_b512_e20_hi1000/c211
fi

if [ ! -f dot-homolog-loss/dot_b256_e20_hi1000/c22/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir dot-homolog-loss/dot_b256_e20_hi1000/c22
fi
if [ ! -f dot-homolog-loss/dot_b256_e20_hi1000/c211/clf_results.pkl ]; then
python -m classifiers.logistic --input_dir dot-homolog-loss/dot_b256_e20_hi1000/c211
fi

##### SUMMARIZING CLASSIFIERS PERFORMANCE #####
unzip baseline_results.zip

python -m viz.baseline_comparison --n_classes 22 --clf_results_dir mse-homolog-loss/mse_b512_e20/c22
python -m viz.baseline_comparison --n_classes 211 --clf_results_dir mse-homolog-loss/mse_b512_e20/c211

python -m viz.baseline_comparison --n_classes 22 --clf_results_dir mse-homolog-loss/mse_b256_e20/c22
python -m viz.baseline_comparison --n_classes 211 --clf_results_dir mse-homolog-loss/mse_b256_e20/c211

python -m viz.baseline_comparison --n_classes 22 --clf_results_dir cross-homolog-loss/cross_b512_e20/c22
python -m viz.baseline_comparison --n_classes 211 --clf_results_dir cross-homolog-loss/cross_b512_e20/c211

python -m viz.baseline_comparison --n_classes 22 --clf_results_dir cross-homolog-loss/cross_b256_e20/c22
python -m viz.baseline_comparison --n_classes 211 --clf_results_dir cross-homolog-loss/cross_b256_e20/c211

python -m viz.baseline_comparison --n_classes 22 --clf_results_dir dot-homolog-loss/dot_b512_e20/c22
python -m viz.baseline_comparison --n_classes 211 --clf_results_dir dot-homolog-loss/dot_b512_e20/c211

python -m viz.baseline_comparison --n_classes 22 --clf_results_dir dot-homolog-loss/dot_b256_e20/c22
python -m viz.baseline_comparison --n_classes 211 --clf_results_dir dot-homolog-loss/dot_b256_e20/c211

python -m viz.baseline_comparison --n_classes 22 --clf_results_dir dot-homolog-loss/dot_b512_e20_hi1000/c22
python -m viz.baseline_comparison --n_classes 211 --clf_results_dir dot-homolog-loss/dot_b512_e20_hi1000/c211

python -m viz.baseline_comparison --n_classes 22 --clf_results_dir dot-homolog-loss/dot_b256_e20_hi1000/c22
python -m viz.baseline_comparison --n_classes 211 --clf_results_dir dot-homolog-loss/dot_b256_e20_hi1000/c211


##### PERFORMN ANOVA AND GET LATEX TABLES #####
python -m utils.stats_reporter --clf_results_dir mse-homolog-loss/mse_b512_e20
python -m utils.stats_reporter --clf_results_dir mse-homolog-loss/mse_b256_e20
python -m utils.stats_reporter --clf_results_dir cross-homolog-loss/cross_b512_e20
python -m utils.stats_reporter --clf_results_dir cross-homolog-loss/cross_b256_e20
python -m utils.stats_reporter --clf_results_dir dot-homolog-loss/dot_b512_e20
python -m utils.stats_reporter --clf_results_dir dot-homolog-loss/dot_b256_e20
python -m utils.stats_reporter --clf_results_dir dot-homolog-loss/dot_b512_e20_hi1000
python -m utils.stats_reporter --clf_results_dir dot-homolog-loss/dot_b256_e20_hi1000
