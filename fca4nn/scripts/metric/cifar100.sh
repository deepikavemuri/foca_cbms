gpu_id=9

CUDA_VISIBLE_DEVICES=$gpu_id python metric_calculator.py \
            --seed 42 \
            --dataset cifar100 \
            --model_name OURS-2FCA::resnet50 \
            --model_weights ./saved_models/extra/cifar100_intsem_2_level_38_0.8604.pt \
            --data_path ./../DATA/cifar100/ \
            --lattice_path ./../DATA/lattices/cifar100_context.pkl \
            --lattice_levels 1 2 \
            --backbone_layer_ids 3 4 \
            --metadata_path ./saved_models/metric_metadata/extra/ \
            --separation_score davies_bouldin \
            --clustering_method kmeans