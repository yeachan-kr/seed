import os
import sys
import argparse
import time
import datetime

import numpy as np
import torch 
import networkx as nx
from tqdm import tqdm

from gensim.models import Word2Vec
from mat2vec.processing import MaterialsTextProcessor

from transformers import AutoModelForTokenClassification, AutoTokenizer
from model.transfer_model import EmbeddingTransfer

mtp = MaterialsTextProcessor()
# mat2vec = Word2Vec.load('./mat_emb/pretrained_embeddings')

# print(mat2vec.wv.most_similar("thermoelectric"))


def main(args):

    # 1. Load down stream dataset
    task_sentences = []
    task_path = os.path.join(args.task_path, args.task_name, 'train.txt')
    with open(task_path, mode='r') as f:
        for line in f:
            line = line[:-1] # remove last special latter (\n)
            task_sentences.append(line)
    print(f'Load {len(task_sentences)} samples from the {args.task_name} dataset (train-set)')


    # 1. Load mat2vec embeddings
    # mtp = MaterialsTextProcessor()
    mat2vec = Word2Vec.load(args.emb_path)
    mat2vec_embs = mat2vec.wv.vectors
    mat2vec_vocab = {}
    for i, w in enumerate(mat2vec.wv.index_to_key):
        mat2vec_vocab[w] = i
    mat2vec_ivocab = dict(zip(mat2vec_vocab.values(), mat2vec_vocab.keys()))
    
    # 2. build frequency vocabulary
    task_vocab = {}
    for line in task_sentences:
        words = line.split(' ')
        for w in words:
            if w not in mat2vec_vocab:
                continue
            if w not in task_vocab:
                task_vocab[w] = 0
            task_vocab[w] += 1
    print(f'All unique words (vocabulay) {len(task_vocab)}')

    # 2. Load language models' vocabulary
    model = AutoModelForTokenClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    lm_vocabs = dict(tokenizer.vocab)
    joint_vocabs = [] 
    for v in mat2vec_vocab:
        if v in lm_vocabs:
            joint_vocabs.append(v)
    disjoint_vocabs = [] 
    for v in task_vocab:
        tokens = tokenizer.tokenize(v)
        if len(tokens) >= args.min_tokens and '_' not in v and v in mat2vec_vocab:
            disjoint_vocabs.append(v)
        
        for (tw, sims) in mat2vec.wv.most_similar(v):
            tokens = tokenizer.tokenize(tw)
            if sims >= 0.7 and len(tokens) >= args.min_tokens and '_' not in tw:
                disjoint_vocabs.append(tw)
    disjoint_vocabs = list(set(disjoint_vocabs))
            # else:

            # if len(tokens) >= args.min_tokens and '_' not in v:
            #     disjoint_vocabs.append(v)
            # if v in task_vocab and len(tokens) >= args.min_tokens:
            #     disjoint_vocabs.append(v)
    print(f'The number of joint words {len(joint_vocabs)}, disjoint words {len(disjoint_vocabs)}')
    # print(len(mat2vec_vocab))
    # exit()
    # 3. Learning transformation matrix (mat2vec to transformer embeddings)
    linear_model = EmbeddingTransfer(input_dims=mat2vec_embs.shape[-1], output_dims=model.bert.embeddings.word_embeddings.weight.size(-1)).cuda()

    source_embeddings = []
    target_embeddings = []
    for w in joint_vocabs:
        source_embeddings.append(torch.from_numpy(mat2vec_embs[mat2vec_vocab[w]]))
        target_embeddings.append(model.bert.embeddings.word_embeddings.weight[lm_vocabs[w]])
    source_embeddings = torch.stack(source_embeddings).cuda()
    target_embeddings = torch.stack(target_embeddings).cuda()
    print(f'Src embedding size {source_embeddings.size()}, Tgt embedding size {target_embeddings.size()}')

    h_loss = torch.nn.HuberLoss(delta=1)
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=1e-3)
    num_iters = int(len(source_embeddings) / args.batch_size)
    for e in range(args.epochs):
        avg_loss = 0.
        num_item = 0.
        randperm = torch.randperm(source_embeddings.size()[0])
        source_embeddings = source_embeddings[randperm]
        target_embeddings = target_embeddings[randperm]
        linear_model.train()
        for i in tqdm(range(num_iters)):
            # print(i)
            src = source_embeddings[i*args.batch_size:(i+1)*args.batch_size].detach()
            tgt = target_embeddings[i*args.batch_size:(i+1)*args.batch_size].detach()
                                    
            transformed_src = linear_model(src)
            
            # sampling two embeddings
            randperm = torch.randperm(transformed_src.size()[0])
            rand_transformed_src = transformed_src[randperm]
            rand_tgt = tgt[randperm]
            
            # relatioal distillation
            r_loss1 = (rand_tgt - tgt) ** 2
            r_loss2 = (rand_transformed_src - transformed_src) ** 2
            r_loss = h_loss(r_loss1, r_loss2).mean()
            
            d_loss = (transformed_src - tgt) ** 2
            
            
            # loss2 = -(torch.softmax(tgt, dim=-1) * torch.log_softmax(transformed_src, dim=-1)).sum(dim=-1).mean()
            # loss2 = (1 - torch.nn.functional.cosine_similarity(transformed_src, tgt)) ** 2
            loss = d_loss.mean() + r_loss #+ loss2.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            num_item += 1
        print(f'Epoch {e}, avg loss {avg_loss/num_item}')

    # transformed embeddings
    # 1. save vocabulary 
    # 2. save embeddings
    disjoint_embeddings = []
    for w in disjoint_vocabs:
        disjoint_embeddings.append(torch.from_numpy(mat2vec_embs[mat2vec_vocab[w]]))
    disjoint_embeddings = torch.stack(disjoint_embeddings).cuda()
    print(f'disjoint embeddings {disjoint_embeddings.size()}')

    linear_model.eval()
    with torch.no_grad():
        transformed_embeddings = linear_model(disjoint_embeddings)
    print('transformed embeddings!')

    import pickle
    # t2e = {}
    # for i, w in enumerate(disjoint_vocabs):
    #     t2e[w] = transformed_embeddings[i]
    pickle.dump(transformed_embeddings, open(f'{args.task_name}.pkl', 'wb'))
    print('save transformed embeddings!')

    lm_ivocabs = dict(zip(lm_vocabs.values(), lm_vocabs.keys()))
    vocab_writer = open(os.path.join(args.output_vocab_path, 'vocab.txt'), 'w')
    for i in range(len(lm_ivocabs)):
        vocab_writer.write(lm_ivocabs[i] + '\n')
    
    for v in disjoint_vocabs:
        vocab_writer.write(v + '\n')
    vocab_writer.close()
    
    # save joint and disjoint dict (key: word, value: embeddings)
    joint_dict = {}
    for i, w in enumerate(joint_vocabs):
        joint_dict[w] = model.bert.embeddings.word_embeddings.weight[lm_vocabs[w]].detach().cpu().numpy()
    disjoint_dict = {}
    for i, w in enumerate(disjoint_vocabs):
        disjoint_dict[w] = transformed_embeddings[i].detach().cpu().numpy()
    print('total words', len(disjoint_dict[w]))
    pickle.dump(joint_dict, open(f'{args.task_name}_joint_dict.pkl', 'wb'))
    pickle.dump(disjoint_dict, open(f'{args.task_name}_disjoint_dict.pkl', 'wb'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # configuration
    parser.add_argument('--task_name', type=str, default='sofc_slot', help='dataset used for vocabulary expansion')
    # parser.add_argument('--task_name', type=str, default='glass_non_glass', help='dataset used for vocabulary expansion')

    # parser.add_argument('--task_name', type=str, default='sofc', help='dataset used for vocabulary expansion')
    # parser.add_argument('--task_name', type=str, default='matscholar', help='dataset used for vocabulary expansion')

    # parser.add_argument('--model_name', type=str, default="allenai/scibert_scivocab_uncased", help='pre-trained language models')
    parser.add_argument('--model_name', type=str, default="bert-base-uncased", help='pre-trained language models')

    parser.add_argument('--threshold', type=float, default=0.6, help='top-k values for the expansion')
    parser.add_argument('--min_tokens', type=int, default=4, help='top-k values for the expansion')
    parser.add_argument('--batch_size', type=int, default=128, help='top-k values for the expansion')
    parser.add_argument('--epochs', type=int, default=50, help='top-k values for the expansion')
    parser.add_argument('--task_path', type=str, default='./downstream_tasks/', help='downstream dataset path (txt format)')
    parser.add_argument('--emb_path', type=str, default='./mat_emb/pretrained_embeddings', help='materials embedding path (gensim bin files)')
    parser.add_argument('--output_vocab_path', type=str, default='./vocabs/', help='output vocabulary path')

    args = parser.parse_args()
    print(args)

    # start processing
    main(args)