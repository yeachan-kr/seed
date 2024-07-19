#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
EC="allenai/scibert_scivocab_uncased" # specify encoder
# EC="bert-base-uncased" # specify encoder
V=1000
LI=5
AT=simclr
CHECKPOINT=checkpoint # specify checkpoint
Data=sofc


for LR in 5e-5
do
  for T in 3
  do
    for fold in 1 2 3 4 5
    do
        for S in 0
        do
            python run_classifier.py \
                --dataset $Data \
                --root data \
                --fold $fold \
                --do_train \
                --seed $S \
                --lr $LR \
                --gradient_accumulation_step 1 \
                --per_gpu_train_batch_size 16 \
                --n_epoch 20 \
                --vocab_size $V \
                --merge_version \
                --temperature $T \
                --transfer_type average_input \
                --evaluate_during_training \
                --checkpoint_dir $CHECKPOINT/${LR} \
                --contrastive \
                --align_type $AT \
                --prototype average \
                --encoder_class $EC ;
        done
        python run_classifier.py \
        --dataset $Data \
        --fold $fold \
        --root data \
        --do_test \
        --lr $LR \
        --seed_list 0 \
        --gradient_accumulation_step 1 \
        --per_gpu_train_batch_size 32 \
        --n_epoch 20 \
        --vocab_size $V \
        --merge_version \
        --contrastive \
        --transfer_type average_input \
        --temperature $T \
        --align_type $AT \
        --prototype average \
        --test_log_dir test_log/${LR} \
        --evaluate_during_training \
        --checkpoint_dir $CHECKPOINT/${LR} \
        --encoder_class $EC ;
    done
  done
done
