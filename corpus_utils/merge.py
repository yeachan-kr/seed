from transformers import BertTokenizer
import pandas as pd
import os
import logging
logger=logging.getLogger(__name__)

## embedding initialization을 위해 추가된 vocabulary와 사전학습된 tokenizer 사이의 mapping을 저장하는 함수 ##
def domain2pretrain(domain_vocab:list,pretrained_tokenizer,vocab_path):

    d2p=dict()
    initial_embedding_id=len(pretrained_tokenizer)

    for embedding_id, key in enumerate(domain_vocab):
        if "##" in key:
            tmp_key = key.replace("##","아")
            values = pretrained_tokenizer.tokenize(tmp_key)
            d2p[key]  =(initial_embedding_id+embedding_id,values[2:], pretrained_tokenizer.convert_tokens_to_ids(values[2:]))
        else:
            values=pretrained_tokenizer.tokenize(key)
            d2p[key] = (initial_embedding_id+embedding_id, values, pretrained_tokenizer.convert_tokens_to_ids(values))

    logger.info("\n Save domain vocab to pretrained vocab mapper %s" %(vocab_path))

    pd.to_pickle(d2p,os.path.join(vocab_path,"d2p.pickle"))

    return d2p

## 사전학습된 모델의 vocabulary에 domain specific subword 추가하는 함수 ##
def merge_domain_vocab(tokenizer,config,domain_vocab:dict,vocab_path):
    '''
    argument:
        tokenizer: domain specific subword가 추가되기 전 tokenizer
        config: 저장할 config
        domain_vocab: 추가할 domain specific subword들
        vocab_path: domain specific subword가 추가된 vocabulary 를 저장할 디렉토리
    '''
    if not os.path.isdir(vocab_path):
        os.makedirs(vocab_path)

    pretrained_vocab=tokenizer.get_vocab()

    new_vocab=[key for key in domain_vocab if key not in pretrained_vocab]

    config.save_pretrained(vocab_path)

    logger.info("\n Merge domain vocab and pretrained vocab at %s" %(vocab_path) )

    f = open(os.path.join(vocab_path, "vocab.txt"), "a")

    for key in new_vocab:
        f.write(key+"\n")
    f.close()

def load_merge_vocab(tokenizer_class,vocab_path):

    return tokenizer_class.from_pretrained(vocab_path)








