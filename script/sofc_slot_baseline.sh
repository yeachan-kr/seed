export CUDA_VISIBLE_DEVICES=$1


EC="allenai/scibert_scivocab_uncased" # specify encoder
CHECKPOINT=checkpoint # specify checkpoint
Data=sofc_slot


for LR in 5e-5
do
    for fold in 1 2 3 4 5
    do
        ## train at three seed
        for S in 0
        do
            python run_classifier.py \
                --dataset $Data \
                --fold $fold \
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
        --fold $fold \
        --root data \
        --do_test \
        --lr $LR \
        --seed_list 0 \
        --n_epoch 20 \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 256 \
        --evaluate_during_training \
        --checkpoint_dir ${CHECKPOINT}/${LR} \
        --test_log_dir test_log/${LR} \
        --encoder_class $EC;
    done
done