# Code for SEED paper

## How to run 

1. Connect mat2vec embeddings with plms (bridge networks training)
~~~
python -u ./connect_mat2vec_to_bert.py  \
    --task 'sofc_slot' \
    --model_name 'bert-base-uncased'
~~~

3. After the base mode training, run the following code for downstraem tasks (task_name = [glass_non_glas, matscholar, soft, sofc_slot])
~~~
bash ./script/[task_name]_avocado.sh
~~~
