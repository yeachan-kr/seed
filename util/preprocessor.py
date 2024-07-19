import os, sys
from transformers import BertTokenizer
from collections import Counter
import csv
from tqdm import tqdm
import pandas as pd


sys.path.append(os.getcwd())
from util.read_dataset import encode_tags
from util.normalize_text import normalize

## Contrastive Learning을 적용하지 않는 NER 데이터셋 ##
class NERPreprocessor(object):
    def __init__(self, data, data_dir, tokenizer: BertTokenizer, seq_len=512, train=False):
        self.seq_len = seq_len
        self.data_dir = data_dir
        self.data = []
        if train:
            labels = Counter()
            for text, label in data:
                self.data.append({'text': text, 'label': label})
                labels += Counter(label)
            labels = dict(labels.most_common()).keys()
            labels = sorted(list(labels))
            self.label_map = {labels[i]: i for i in range(len(labels))}
            pd.to_pickle(self.label_map, os.path.join(self.data_dir, "label.json"))
        else:
            for text, label in data:
                self.data.append({'text': text, 'label': label})
            self.label_map = pd.read_pickle(os.path.join(self.data_dir, "label.json"))
        
        self.df = pd.DataFrame(self.data)
        self.tokenizer = tokenizer

    def parse(self):
        X, y = self.remove_zero_len_tokens()
        self.df["lens"] = [len(t) for t in X]
        encodings = self.tokenizer(X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=self.seq_len)
        labels = encode_tags(self.label_map, y, encodings)
        encodings.pop('offset_mapping')
        res = []
        keys = encodings.keys()
        for i, label in tqdm(enumerate(labels), total=len(labels)):
            data = {key: encodings[key][i] for key in keys}
            data['label'] = label
            res.append(data)

        return pd.DataFrame(res)
    
    def remove_zero_len_tokens(self):
        X, y = [], []
        for sent, labels in zip(self.df['text'], self.df['label']):
            new_sent, new_labels = [], []
            for token, label in zip(sent, labels):
                if len(self.tokenizer.tokenize(token)) == 0:
                    continue
                new_sent.append(token)
                new_labels.append(label)
            X.append(new_sent)
            y.append(new_labels)
        return X, y

## Contrastive Learning을 적용하는 NER 데이터셋 ##
class ContrastiveNERPreprocessor(NERPreprocessor):
    def __init__(self, data, data_dir, tokenizer: BertTokenizer, domain_tokenizer, seq_len=512, train=False):
        super(ContrastiveNERPreprocessor, self).__init__(data, data_dir, tokenizer, seq_len, train)
        self.domain_tokenizer = domain_tokenizer

    def parse(self):
        X, y = self.remove_zero_len_tokens()
        self.df["lens"] = [len(t) for t in X]
        encodings = self.tokenizer(X, is_split_into_words=True, return_offsets_mapping=False, padding=True, truncation=True, max_length=self.seq_len)
        domain_encodings = self.domain_tokenizer(X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=self.seq_len)
        labels = encode_tags(self.label_map, y, domain_encodings)
        domain_encodings.pop('offset_mapping')
        res = []
        keys = encodings.keys()
        for i, label in tqdm(enumerate(labels), total=len(labels)):
            data = {key: encodings[key][i] for key in keys}
            for key in domain_encodings.keys():
                data[f'domain_{key}'] = domain_encodings[key][i]
            data['label'] = label
            res.append(data)
        return pd.DataFrame(res)
    
    def remove_zero_len_tokens(self):
        X, y = [], []
        for sent, labels in zip(self.df['text'], self.df['label']):
            new_sent, new_labels = [], []
            for token, label in zip(sent, labels):
                if len(self.domain_tokenizer.tokenize(token)) == 0:
                    continue
                new_sent.append(token)
                new_labels.append(label)
            X.append(new_sent)
            y.append(new_labels)
        return X, y


## Contrastive Learning을 적용하지 않는 Classification 데이터셋 ##
class CLSPreprocessor(object):
    def __init__(self, data_dir, data_name, tokenizer: BertTokenizer, train=False):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, data_name), newline='') as reader:
            lines = csv.reader(reader, delimiter=',')
            self.data=[]
            for i, line in enumerate(lines):
                if i==0: continue
                text = normalize(line[0])
                self.data.append({'text': text, 'label': line[1]})
        self.df = pd.DataFrame(self.data)
        self.tokenizer = tokenizer

        if train:
            label = self.df.label.value_counts().keys()
            self.label_map = {label[i]: i for i in range(len(label))}
            pd.to_pickle(self.label_map, os.path.join(self.data_dir, "label.json"))
        else:
            self.label_map = pd.read_pickle(os.path.join(self.data_dir, "label.json"))

    def parse(self):
        self.df["lens"] = [len(t) for t in self.df["text"].to_list()]
        res = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            encodings = self._tokenize(row['text'])
            data = {key: encodings[key] for key in encodings.keys()}
            data['label'] = self.label_map[row['label']]
            res.append(data)

        return pd.DataFrame(res)

    def _tokenize(self, text):

        return self.tokenizer(text)

    def _tokenize_exbert(self, text):

        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)


## Contrastive Learning을 적용하지 않는 Classification 데이터셋 ##
class ContrastiveCLSPreprocessor(CLSPreprocessor):
    def __init__(self, data_dir, data_name, tokenizer: BertTokenizer, domain_tokenizer, train=False):
        super(ContrastiveCLSPreprocessor, self).__init__(data_dir, data_name, tokenizer, train)
        self.domain_tokenizer = domain_tokenizer

    def parse(self):
        res = []
        self.df["lens"] = [len(t) for t in self.df["text"].to_list()]

        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            encodings = self.tokenizer(row['text'])
            domain_encodings = self.domain_tokenizer(row['text'])
            data = {key: encodings[key] for key in encodings.keys()}
            for key in domain_encodings.keys():
                data[f'domain_{key}'] = domain_encodings[key]
            data['label'] = self.label_map[row['label']]
            res.append(data)

        return pd.DataFrame(res)

    def _tokenize(self, text):
        return self.tokenizer(text)