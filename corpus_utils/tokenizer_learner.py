from transformers import BertTokenizer

import os
from .merge import load_merge_vocab
import logging
import pandas as pd

logger = logging.getLogger(__name__)

## Domain specific subwords를 추가하는 클래스 ##
class Learner:
    '''
    argument:
        args: Argument Parser
        config: 사전학습된 모델의 configure
        pretrain_tokenizer: AutoTokenizer (사전학습된 tokenizer, 그리고 확장된 vocabulary를 적용할 tokenizer)
        domain_tokenizer: bpe_mapper.py 의 CustomTokenizer (추가될 subword를 생성하는 클래스)
    '''
    def __init__(self, args, config, pretrain_tokenizer, domain_tokenizer, init_feritility=2):
        self.args = args
        self.config = config

        self.vocab_path = self.args.vocab_path
        self.pretrain_tokenizer = pretrain_tokenizer
        self.domain_tokenizer = domain_tokenizer
        self.pretrain_tokenizer.save_pretrained(self.vocab_path)

        self.unique_corpus = None
        self.init_fertility = init_feritility

        config.save_pretrained(self.vocab_path)

    def init_long_corpus(self, unique_corpus, tokenizer: BertTokenizer):
        out = []
        for w in unique_corpus:
            tokens = tokenizer.tokenize(w)
            if len(tokens) > self.init_fertility:
                out.append(w)
        self.unique_corpus = out

    ## subword를 추가할지 말지 결정하는 score 계산 ##
    def compute_fertility(self, unique_corpus: list, tokenizer: BertTokenizer):
        nominator = []
        for w in unique_corpus:
            nominator.extend(tokenizer.tokenize(w))

        return len(nominator) / len(unique_corpus)

    ## fertility score 기반으로 subword를 추가하는 함수 ##
    def update_tokenizer(self, unique_words, n_chunk=50):

        pretrain_vocab = self.pretrain_tokenizer.get_vocab()
        domain_vocab = self.domain_tokenizer.get_vocab().items()
        ps = sorted(domain_vocab, key=lambda x: x[-1])

        init = 500 # 500개부터 추가
        candidate_vocab = [k for k, _ in ps if k not in pretrain_vocab] # 사전학습된 모델의 vocabulary와 겹치지 않는 vocabulary 후보 선정
        for_save = []
        init_func = self.init_long_corpus
        update_func = self.compute_fertility

        init_func(unique_words, self.pretrain_tokenizer)
        F = update_func(self.unique_corpus, self.pretrain_tokenizer) # fertility 계산
        for_save.append(F)
        logger.info("Initial fertility {0:.6f} ".format(F))
        remains = candidate_vocab
        step = 0

        while F > 3.0 and len(remains) > 0: # fertility score가 3 미만이 되거나 candidate_vocab에서 추가할 vocabulary가 더이상 없는 경우
            step += 1
            # step마다 추가할 vocabulary 수가 n_chunk씩 증가, 이전 step에서 추가한 vocabulary는 후보군(remains)에서 제외
            domain_one, remains = remains[:init + n_chunk], remains[init + n_chunk:] 
            self.pretrain_tokenizer = self.add_domain_vocab(domain_vocab=domain_one)
            F = update_func(self.unique_corpus, self.pretrain_tokenizer)
            logger.info("Current fertility {0:.10f} ".format(F))
            init += n_chunk
            for_save.append(F)
        print(init)
        pd.to_pickle(F, os.path.join(self.vocab_path, "feritility"))

        return candidate_vocab[:init]

    ## Vocabulary 경로에 subword를 추가하는 함수 ##
    def add_domain_vocab(self, domain_vocab: str):

        if not os.path.isdir(self.vocab_path):
            os.makedirs(self.vocab_path)

        f = open(os.path.join(self.vocab_path, "vocab.txt"), "a")

        for vocab in domain_vocab:
            f.write(vocab + "\n")
        f.close()

        return load_merge_vocab(tokenizer_class=self.pretrain_tokenizer, vocab_path=self.vocab_path)