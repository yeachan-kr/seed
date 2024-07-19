import pandas as pd
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
import random

from torch.nn.utils.rnn import pad_sequence

## Dataset의 parent 클래스 ##
class Base_Batchfier(IterableDataset):
    def __init__(self, args, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 512,
                 criteria: str = 'lens', padding_index=70000, epoch_shuffle=True, device='cuda'):
        super(Base_Batchfier).__init__()
        self.args = args
        self.maxlen = maxlen
        self.minlen = minlen
        self.size = batch_size
        self.criteria = criteria
        self.seq_len = seq_len
        self.padding_index = padding_index
        self.epoch_shuffle = epoch_shuffle
        self.device = device

    def truncate_small(self, df, criteria='lens'):
        lens = np.array(df[criteria])
        indices = np.nonzero((lens < self.minlen).astype(np.int64))[0]
        return df.drop(indices)

    def truncate_large(self, texts, lens):
        new_texts = []
        new_lens = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > self.maxlen:
                new_texts.append(text[:self.maxlen])
                new_lens.append(self.maxlen)
            else:
                remainder = len(text) % self.seq_len
                l = lens[i]
                if remainder and remainder < 10:
                    text = text[:-remainder]
                    l = l - remainder
                new_texts.append(text)
                new_lens.append(l)
        return new_texts, new_lens

    def shuffle(self, df, num_buckets):
        dfs = []
        for bucket in range(num_buckets - 1):
            new_df = df.iloc[bucket * self.size: (bucket + 1) * self.size]
            dfs.append(new_df)
        random.shuffle(dfs)
        dfs.append(df.iloc[num_buckets - 1 * self.size: num_buckets * self.size])
        df = pd.concat(dfs)
        return df

## Contrastive Learning을 적용하지 않은 Dataset ##
class CFBatchFier(Base_Batchfier):
    def __init__(self, args, df: pd.DataFrame, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 512,
                 criteria: str = 'lens', padding_index=0, epoch_shuffle=True, device='cuda'):
        super(CFBatchFier, self).__init__(args, batch_size, seq_len, minlen, maxlen, criteria, padding_index,
                                          epoch_shuffle,
                                          device)

        self.size = batch_size
        self.df = df
        self.df["lens"] = [len(text) for text in self.df.input_ids]
        self.num_buckets = len(self.df) // self.size + (len(self.df) % self.size != 0)
        self.df = self.sort(self.df, criteria="lens")

        if epoch_shuffle:
            self.df = self.shuffle(self.df, self.num_buckets)

    def _maxlens_in_first_batch(self, df):
        first_batch = df.iloc[0:self.size]

        return first_batch

    def shuffle(self, df, num_buckets):
        dfs = []
        for bucket in range(1, num_buckets):
            new_df = df.iloc[bucket * self.size: (bucket + 1) * self.size]
            dfs.append(new_df)

        random.shuffle(dfs, )
        first_batch = self._maxlens_in_first_batch(df)
        dfs.insert(0, first_batch)
        df = pd.concat(dfs)

        return df.reset_index(drop=True)

    def sort(self, df, criteria="lens"):
        return df.sort_values(criteria, ascending=False).reset_index(drop=True)

    def truncate_large(self):
        lens = np.array(self.df["lens"])
        indices = np.nonzero((lens > self.maxlen).astype(np.int64))[0]

        self.df = self.df.drop(indices).reset_index(drop=True)

    def __iter__(self):
        for _, row in self.df.iterrows():
            data = {key: row[key][:self.maxlen] for key in row.keys() if key not in ['label', 'lens']}
            if isinstance(row["label"],list):
                data['label'] = row['label'][:self.maxlen]
                yield data
            else:
                data['label'] = row['label']
                yield data

    def __len__(self):
        return self.num_buckets



    def collate(self, batch):
        batched = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'label': []}
        for key in batched.keys():
            if key == 'label':
                batched[key] = [torch.LongTensor([item[key]]) for item in batch]
                batched[key] = pad_sequence(batched[key])
            else:
                batched[key] = [torch.LongTensor(item[key]) for item in batch]
                batched[key] = pad_sequence(batched[key], batch_first=True, padding_value=self.padding_index)
        return batched


    def collate_ner(self, batch):
        batched = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'label': []}
        for key in batched.keys():
            if key == 'label':
                batched[key] = [torch.LongTensor(item[key]) for item in batch]
                batched[key] = pad_sequence(batched[key], batch_first=True, padding_value=-100)
            else:
                batched[key] = [torch.LongTensor(item[key]) for item in batch]
                batched[key] = pad_sequence(batched[key], batch_first=True, padding_value=self.padding_index)
                
        return batched

## Contrastive Learning을 적용하기 위한 Dataset ##
class ContrastiveBatchFier(CFBatchFier):
    def __init__(self, args, df: pd.DataFrame, batch_size: int = 32, seq_len=1024, minlen=50, maxlen: int = 512,
                 criteria: str = 'lens', padding_index=0, epoch_shuffle=False, device='cuda', mask_idx=-1,masked_prob=0.1):
        super(ContrastiveBatchFier, self).__init__(args, df, batch_size, seq_len, minlen, maxlen, criteria,
                                                   padding_index,
                                                   epoch_shuffle, device)
        self.mask_prob = masked_prob
        self.mask_idx = mask_idx

    def __iter__(self):
        for _, row in self.df.iterrows():
            data = {key: row[key][:self.maxlen] for key in row.keys() if key not in ['label', 'lens']}
            if isinstance(row["label"],list):
                data['label'] = row['label'][:self.maxlen]
                yield data
            else:
                data['label'] = row['label']
                yield data
                

    def __len__(self):
        return self.num_buckets

    def collate(self, batch):
        batched = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 
                   'domain_input_ids': [], 'domain_token_type_ids': [], 'domain_attention_mask': [],
                   'label': []}
        for key in batched.keys():
            if key == 'label':
                batched[key] = [torch.LongTensor([item[key]]) for item in batch]
                batched[key] = pad_sequence(batched[key])
            else:
                batched[key] = [torch.LongTensor(item[key]) for item in batch]
                batched[key] = pad_sequence(batched[key], batch_first=True, padding_value=self.padding_index)
        return batched


    def collate_ner(self, batch):
        batched = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 
                   'domain_input_ids': [], 'domain_token_type_ids': [], 'domain_attention_mask': [],
                   'label': []}
        for key in batched.keys():
            if key == 'label':
                batched[key] = [torch.LongTensor(item[key]) for item in batch]
                batched[key] = pad_sequence(batched[key], batch_first=True, padding_value=-100)
            else:
                batched[key] = [torch.LongTensor(item[key]) for item in batch]
                batched[key] = pad_sequence(batched[key], batch_first=True, padding_value=self.padding_index)
                
        return batched