import os
from .preprocessor import *
from transformers import BertTokenizer

import logging
from .read_dataset import get_ner_data

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


## Raw 데이터셋 불러오기 ##
def get_dataset(args, tokenizer: BertTokenizer):
    res = {}
    data_dir = os.path.join(args.root, args.dataset)
    if args.fold is not None:
        data_dir = os.path.join(data_dir, str(args.fold))
    cache_dir = os.path.join(data_dir, "cache")

    if args.merge_version:
        cache_dir += "_merged"
    elif args.other:
        cache_dir += "_other"

    dataset_dir = os.path.join(cache_dir, args.encoder_class)

    if args.contrastive:
        dataset_dir += "-contrastive-"

    if args.merge_version:
        dataset_dir += "{0}_optimized".format(args.vocab_size)


    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    if args.dataset == "glass_non_glass":
        data_list = ["train", "val", "test"]
        for d_name in data_list:
            train = "train" == d_name

            if args.contrastive:
                assert isinstance(tokenizer, tuple)
                pretrained_tokenizer, domain_tokenizer = tokenizer
                builder = ContrastiveCLSPreprocessor(data_dir, f"{d_name}.csv", pretrained_tokenizer, domain_tokenizer,
                                                    train)
            else:
                builder = CLSPreprocessor(data_dir, f"{d_name}.csv", tokenizer, train)

            dataset = builder.parse()
            res[d_name] = dataset
        
        res["label"] = builder.label_map
        return res["train"], res["val"], res["test"], res["label"]
    
    else:
        data_list = ["train", "dev", "test"]
        raw_data = get_ner_data(args.dataset, args.fold, norm=True)
        split_data = {'train': zip(raw_data[0], raw_data[1]), 'dev': zip(raw_data[2], raw_data[3]), 'test': zip(raw_data[4], raw_data[5])}
        for d_name in data_list:
            train = "train" == d_name

            if args.contrastive:
                assert isinstance(tokenizer, tuple)
                pretrained_tokenizer, domain_tokenizer = tokenizer
                builder = ContrastiveNERPreprocessor(split_data[d_name], dataset_dir, pretrained_tokenizer, domain_tokenizer,
                                                    args.seq_len, train)
            else:
                builder = NERPreprocessor(split_data[d_name], dataset_dir, tokenizer, args.seq_len, train)

            dataset = builder.parse()
            res[d_name] = dataset

        res["label"] = builder.label_map
        return res["train"], res["dev"], res["test"], res["label"]

