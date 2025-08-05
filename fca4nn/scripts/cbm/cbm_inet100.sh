gpu_id=4

CUDA_VISIBLE_DEVICES=$gpu_id python train_cbm.py \
                --do_train \
                --do_test \
                --seed 42 \
                --dataset imagenet100 \
                --model resnet50 \
                --concept_wts 0.01 \
                --data_dir /raid/DATASETS/inet100 \
                --concept_file ./concept_project/data/concepts/inet100_concepts.json \
                --lr 1e-4 \
                --epochs 3 \
                --batch_size 256 \
                --verbose 10 