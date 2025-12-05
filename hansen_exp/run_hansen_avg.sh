for donor in 10021 9861; do
    for dataset in hansen_full hansen_recommended; do
        echo "Processing donor $donor dataset $dataset"
        python avg_inr_atlas.py \
            --mode avg_atlas \
            --donor $donor \
            --gene_list ./data/abagendata/train_${dataset}/se_${donor}.csv \
            --input_dir ./results/${dataset}_${donor} \
            --all_records
    done
done