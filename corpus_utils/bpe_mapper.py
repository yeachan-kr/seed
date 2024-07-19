import os
import pandas as pd
from tqdm import tqdm
import re
import csv
import logging
from tokenizers import BertWordPieceTokenizer

from util.read_dataset import get_ner_data

logger = logging.getLogger(__name__)


class CustomTokenizer:
    '''
    argument:
        args: Argument Parser(
            encoder_class: 사전학습된 모델 (bert-base-uncased / m3rg-iitd/matscibert)
            root: 데이터셋이 저장된 디렉토리,
            vocab_size: domain specific corpus에서 새로 생성할 vocab의 크기,
                (단, vocab expansion 시 추가되는 vocab 크기와 다름)
            dataset: Vocab Expansion을 적용할 downstream task ['sofc_slot', 'matscholar']
        )
    '''
    def __init__(self, args, train_file_path):
        self.args = args
        self.vocab_path = args.vocab_path
        self.dir_path = args.root
        self.prefix = args.dataset
        self.vocab_size = args.vocab_size
        
        self.train_path = train_file_path

        self.encoder = self.load_encoder()

    ## 가공되지 않은 cls data 중 train 파일을 text파일로 저장 ##
    def save_cls_txt(self):

        data_dir = os.path.join(self.dir_path, self.prefix)

        logger.info("Flatten Corpus to text file")
        with open(os.path.join(data_dir, 'train.csv'), newline='') as reader:
            lines = csv.reader(reader, delimiter=',')
            self.data=[]
            for i, line in enumerate(lines):
                if i==0: continue
                self.data.append({'text': line[0], 'label': line[1]})
        df = pd.DataFrame(self.data)

        txt_file = os.path.join(data_dir, "train.txt")
        f = open(txt_file, "w")
        textlines = []

        for i, row in df.iterrows():
            new_string = re.sub('[^a-zA-Z0-9\n\.]', ' ', row["text"])
            new_string = re.sub(' +', ' ', new_string)
            textlines.append(new_string.replace("\n", " "))

        for textline in tqdm(textlines):
            f.write(textline + "\n")

    
    ## 가공되지 않은 ner data 중 train 파일을 text파일로 저장 ##
    def save_ner_txt(self):
        data_dir = os.path.join(self.dir_path, self.prefix)

        logger.info("Flatten Corpus to text file")
        if self.args.encoder_class == 'm3rg-iitd/matscibert': # matscibert의 경우 normalization 적용
            raw_data = get_ner_data(self.prefix, None, norm=True) # raw_data: train_X, train_y, dev_X, dev_y, test_X, test_y로 구성됨
        else:
            raw_data = get_ner_data(self.prefix, None, norm=False)
            
        split_data = {'train': zip(raw_data[0], raw_data[1]), 'dev': zip(raw_data[2], raw_data[3]), 'test': zip(raw_data[4], raw_data[5])}
        
        self.data=[]
        for text, label in split_data['train']:
            self.data.append({'text': text, 'label': label})
            
        if self.prefix == "sofc_slot": # SOFC-Slot 데이터셋의 경우 train셋과 dev셋을 합쳐서 vocab 추출 (fold에 따라 train, dev 셋이 달라지기 때문)
            for text, label in split_data['dev']:
                self.data.append({'text': text, 'label': label})
                
        df = pd.DataFrame(self.data)

        os.makedirs(data_dir, exist_ok=True)
        txt_file = os.path.join(data_dir, "train.txt")
        f = open(txt_file, "w")
        textlines = []

        for i, row in df.iterrows():
            new_string = re.sub('[^a-zA-Z0-9\n\.]', ' ', ' '.join(row["text"]))
            new_string = re.sub(' +', ' ', new_string)
            textlines.append(new_string.replace("\n", " "))

        for textline in tqdm(textlines):
            f.write(textline + "\n")

    ## Wordpiece Tokenizer 학습 ##
    def train(self):
        # if self.prefix == "glass_non_glass":
        #     self.save_cls_txt()
        # else:
        #     self.save_ner_txt()
        # # txt_path = os.path.join(self.dir_path, self.prefix, "train.txt")\

        self.encoder.train(self.train_path, vocab_size=self.vocab_size, min_frequency=1)
        self.encoder.save_model(directory=self.vocab_path, prefix="{0}_{1}".format(self.prefix, str(self.vocab_size)))

    ## Domain specific Wordpiece Tokenizer 불러오기 ##
    def load_encoder(self):
        self.encoder = BertWordPieceTokenizer(lowercase=True)
        base_name = os.path.join(self.vocab_path, "{0}_{1}".format(self.prefix, self.vocab_size))
        vocab_name = base_name + '-vocab.txt'

        # ## Domain specific vocab 존재하면 불러오기
        # if os.path.exists(vocab_name):
        #     logger.info('\ntrained encoder loaded')
        #     return BertWordPieceTokenizer.from_file(vocab_name)
        
        # ## Domain specific vocab 존재하지 않으면 새로 Wordpiece Tokenizer 학습
        # else:
        logger.info('\nencoder needs to be trained')
        self.train()
        return self.encoder
        