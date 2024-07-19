#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
EC=bert-base-uncased # specify encoder
V=1000
LI=5
AT=simclr
CHECKPOINT=checkpoint # specify checkpoint
Data=glass_non_glass

for LR in 5e-5
do
  ## train at three seed
  for S in 0 
  do
    python run_classifier.py \
      --dataset $Data \
      --root data \
      --do_train \
      --seed $S \
      --lr $LR \
      --gradient_accumulation_step 2 \
      --per_gpu_train_batch_size 8 \
      --n_epoch 20 \
      --vocab_size $V \
      --merge_version \
      --transfer_type average_input \
      --evaluate_during_training \
      --checkpoint_dir $CHECKPOINT/${LR} \
      --align_type $AT \
      --prototype average \
      --encoder_class $EC ;
  done
  python run_classifier.py \
  --dataset $Data \
  --root data \
  --do_test \
  --lr $LR \
  --seed_list 0 1 2 \
  --gradient_accumulation_step 2 \
  --per_gpu_train_batch_size 8 \
  --n_epoch 10 \
  --vocab_size $V \
  --merge_version \
  --transfer_type average_input \
  --align_type $AT \
  --prototype average \
  --test_log_dir test_log/${LR} \
  --evaluate_during_training \
  --checkpoint_dir $CHECKPOINT/${LR} \
  --encoder_class $EC ;
done