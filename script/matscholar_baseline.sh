EC="bert-base-uncased"
CHECKPOINT=checkpoint # specify checkpoint
Data=matscholar

for LR in 5e-5
do
  ## train at three seed
  for S in 1
  do
  python run_classifier.py \
      --dataset $Data \
      --root data \
      --do_train \
      --seed $S \
      --lr $LR \
      --gradient_accumulation_step 1 \
      --per_gpu_train_batch_size 16 \
      --n_epoch 20 \
      --evaluate_during_training \
      --checkpoint_dir ${CHECKPOINT}/${LR} \
      --encoder_class $EC ;
  done
  python run_classifier.py \
  --dataset $Data \
  --root data \
  --do_test \
  --lr $LR \
  --seed_list 1 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 256 \
  --evaluate_during_training \
  --checkpoint_dir ${CHECKPOINT}/${LR} \
  --test_log_dir test_log/${LR} \
  --encoder_class $EC;
done