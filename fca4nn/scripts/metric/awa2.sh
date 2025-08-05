gpu_id=14

CUDA_VISIBLE_DEVICES=$gpu_id python metric_calculator.py \
            --seed 42 \
            --dataset awa2 \
            --model_name OURS-2FCA::resnet18 \
            --model_weights ./saved_models/extra/awa2_intsem_2_level_99_0.9768.pt \
            --data_path ./../DATA/Animals_with_Attributes2/ \
            --lattice_path ./../DATA/lattices/awa2_context.pkl \
            --lattice_levels 1 3 \
            --backbone_layer_ids 3 4 \
            --metadata_path ./saved_models/metric_metadata/extra/ \
            --separation_score davies_bouldin \
            --clustering_method kmeans