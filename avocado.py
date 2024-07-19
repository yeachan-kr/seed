from corpus_utils.bpe_mapper import CustomTokenizer
from util.args import CorpusArgument
import logging
from transformers import AutoTokenizer, AutoConfig
from corpus_utils.merge import domain2pretrain
from corpus_utils.tokenizer_learner import Learner
import re
import os

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    args = CorpusArgument()

    domain_tokenizer = CustomTokenizer(args)

    pretrained_tokenizer = AutoTokenizer.from_pretrained(args.encoder_class)
    pretrained_tokenizer.save_pretrained(args.vocab_path)
    pretrained_config = AutoConfig.from_pretrained(args.encoder_class)

    txt_file = os.path.join(args.root, args.dataset, "train.txt")
    with open(txt_file, "r") as f:
        out = f.read()
    new_string = re.sub('[^a-zA-Z0-9\n\.]', ' ', out)
    new_string = re.sub(' +', ' ', new_string)

    unique_words = list(new_string.strip().replace("\n", " ").split(" "))

    learner = Learner(args, pretrained_config, pretrained_tokenizer, domain_tokenizer.encoder, ) 
    added_vocab = learner.update_tokenizer(unique_words, 50)
    # print('add', added_vocab, len(added_vocab))
    # exit()

    d2p = domain2pretrain(added_vocab, pretrained_tokenizer, vocab_path=args.vocab_path) 
