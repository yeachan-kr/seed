#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

EC="bert-base-uncased"
V=1000

for Data in glass_non_glass
do
  python avocado.py --dataset $Data \
    --root data \
    --vocab_size $V \
    --encoder_class $EC;
done