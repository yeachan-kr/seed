EC=m3rg-iitd/matscibert # specify encoder
CHECKPOINT=checkpoint # specify checkpoint
Data=glass_non_glass

for LR in 2e-5 3e-5 5e-5
do
  ## train at three seed
  for S in 0 1 2
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
      --evaluate_during_training \
      --checkpoint_dir ${CHECKPOINT}/${LR} \
      --encoder_class $EC ;
  done
  python run_classifier.py \
  --dataset $Data \
  --root data \
  --do_test \
  --lr $LR \
  --seed_list 0 1 2 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 256 \
  --evaluate_during_training \
  --checkpoint_dir ${CHECKPOINT}/${LR} \
  --test_log_dir test_log/${LR} \
  --encoder_class $EC;
done