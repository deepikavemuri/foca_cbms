gpu_id=12

CUDA_VISIBLE_DEVICES=$gpu_id python metric_calculator.py \
            --seed 42 \
            --dataset inet100 \
            --model_name OURS-2FCA::resnet50 \
            --model_weights ./saved_models/extra/inet100_intsem_2_level_29_0.9129.pt \
            --data_path /raid/DATASETS/inet100/ \
            --lattice_path ./../DATA/lattices/inet100_context.pkl \
            --lattice_levels 1 3\
            --backbone_layer_ids 3 4 \
            --metadata_path ./saved_models/metric_metadata/extra/ \
            --separation_score davies_bouldin \
            --clustering_method kmeans