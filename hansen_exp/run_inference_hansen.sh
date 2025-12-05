#!/bin/bash

# Case 1: Hansen Full, Donor 10021
echo "Running Case 1: Hansen Full, Donor 10021"
python inference_new.py \
    --model_path ./hansen_ckpts/siren_hansen_full_10021_0.0001_27x512x12_1.5844907851059187e-12.pth \
    --donor 10021 \
    --dataset hansen_full \
    --gene_list ./data/abagendata/train_hansen_full/se_10021.csv \
    --output_dir ./results/hansen_full_10021 \
    --all_records \
    --hidden_features 512 \
    --hidden_layers 12 \
    --in_features 27

# Case 2: Hansen Full, Donor 9861
echo "Running Case 2: Hansen Full, Donor 9861"
python inference_new.py \
    --model_path ./hansen_ckpts/siren_hansen_full_9861_0.0001_27x512x12_2.8973248800134854e-10.pth \
    --donor 9861 \
    --dataset hansen_full \
    --gene_list ./data/abagendata/train_hansen_full/se_9861.csv \
    --output_dir ./results/hansen_full_9861 \
    --all_records \
    --hidden_features 512 \
    --hidden_layers 12 \
    --in_features 27

# Case 3: Hansen Recommended, Donor 10021
echo "Running Case 3: Hansen Recommended, Donor 10021"
python inference_new.py \
    --model_path ./hansen_ckpts/siren_hansen_recommended_10021_0.0001_27x512x12_1.5219948782188575e-12.pth \
    --donor 10021 \
    --dataset hansen_recommended \
    --gene_list ./data/abagendata/train_hansen_recommended/se_10021.csv \
    --output_dir ./results/hansen_recommended_10021 \
    --all_records \
    --hidden_features 512 \
    --hidden_layers 12 \
    --in_features 27

# Case 4: Hansen Recommended, Donor 9861
echo "Running Case 4: Hansen Recommended, Donor 9861"
python inference_new.py \
    --model_path ./hansen_ckpts/siren_hansen_recommended_9861_0.0001_27x512x12_1.6911900482460829e-12.pth \
    --donor 9861 \
    --dataset hansen_recommended \
    --gene_list ./data/abagendata/train_hansen_recommended/se_9861.csv \
    --output_dir ./results/hansen_recommended_9861 \
    --all_records \
    --hidden_features 512 \
    --hidden_layers 12 \
    --in_features 27