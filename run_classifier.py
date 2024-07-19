from util.ner_trainer import CFNERTrainer, ContrastiveNERTrainer
from util.cls_trainer import CFCLSTrainer, ContrastiveCLSTrainer

from util.data_builder import get_dataset
from util.args import ExperimentArgument
from tqdm import tqdm
import pandas as pd
from embedding_utils.embedding_initializer import transfer_embedding
from transformers import AdamW
from util.batch_generator import CFBatchFier, ContrastiveBatchFier
from model.classification_model import ModelForTokenClassification, ModelForSequenceClassification

from transformers import get_scheduler
import pickle

import torch
import random
from util.logger import *
import logging
import math

logger = logging.getLogger(__name__)


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_trainer(args, model, train_batchfier, test_batchfier):
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not 'bert' in n], 'lr': 3e-4},
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.lr}
    ]
    optimizer_kwargs = {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    }
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)


    num_update_steps_per_epoch = len(train_batchfier) // args.gradient_accumulation_step
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_training_steps = math.ceil(args.n_epoch * num_update_steps_per_epoch)
    warmup_steps = math.ceil(num_training_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    if args.dataset == "glass_non_glass":
        if args.contrastive:
            trainer = ContrastiveCLSTrainer(args, model, train_batchfier, test_batchfier, optimizer, scheduler,
                                        args.gradient_accumulation_step, args.clip_norm, args.n_label)
        else:
            trainer = CFCLSTrainer(args, model, train_batchfier, test_batchfier, optimizer, scheduler,
                                args.gradient_accumulation_step, args.clip_norm, args.n_label)
    else:
        if args.contrastive:
            trainer = ContrastiveNERTrainer(args, model, train_batchfier, test_batchfier, optimizer, scheduler,
                                        args.gradient_accumulation_step, args.clip_norm, args.n_label)
        else:
            trainer = CFNERTrainer(args, model, train_batchfier, test_batchfier, optimizer, scheduler,
                                args.gradient_accumulation_step, args.clip_norm, args.n_label)

    return trainer

## Tokenized Dataset ##
def get_batchfier(args, tokenizer):
    n_gpu = torch.cuda.device_count()
    train, dev, test, label = get_dataset(args, tokenizer)
    if isinstance(tokenizer, tuple):
        _, domain_tokenizer = tokenizer
        padding_idx = domain_tokenizer.pad_token_id
        mask_idx = domain_tokenizer.pad_token_id
    else:
        padding_idx = tokenizer.pad_token_id
        mask_idx = tokenizer.pad_token_id
    if args.contrastive: # Contrastive Learning을 적용하는 데이터셋
        train_batch = ContrastiveBatchFier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu,
                                           maxlen=args.seq_len,
                                           padding_index=padding_idx, mask_idx=mask_idx)
        dev_batch = ContrastiveBatchFier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu,
                                         maxlen=args.seq_len, epoch_shuffle=False,
                                         padding_index=padding_idx, mask_idx=mask_idx)
        test_batch = ContrastiveBatchFier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu,
                                          maxlen=args.seq_len, epoch_shuffle=False,
                                          padding_index=padding_idx, mask_idx=mask_idx)

    else: # Contrastive Learning을 적용하지 않는 데이터셋
        train_batch = CFBatchFier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu, maxlen=args.seq_len,
                                  padding_index=padding_idx)
        dev_batch = CFBatchFier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                                epoch_shuffle=False, padding_index=padding_idx)
        test_batch = CFBatchFier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                                 epoch_shuffle=False, padding_index=padding_idx)

    return train_batch, dev_batch, test_batch, label

## Embedding Size 조정 ##
def expand_token_embeddings(model, tokenizer):
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_num_tokens=new_vocab_size)

## Embedding 초기화 ##
def embedding(args, model, d2p):
    transfer_embedding(model, d2p, args.transfer_type)


def main():
    args = ExperimentArgument()
    args.aug_ratio = 0.0
    set_seed(args.seed)
    gpu = 0
    args.gpu = gpu
    
    if args.dataset == "glass_non_glass":
        args.n_epoch = 10
    elif args.dataset == "sofc":
        args.n_epoch = 20
    elif args.dataset == "matscholar":
        args.n_epoch = 15
    else:
        args.n_epoch = 40
    
    from transformers import AutoConfig,AutoTokenizer
    pretrained_config = AutoConfig.from_pretrained(args.encoder_class)

    if args.merge_version:
        tokenizer = AutoTokenizer.from_pretrained(args.vocab_path)
        pretrained_tokenizer = AutoTokenizer.from_pretrained(args.encoder_class)

        # read additional tokens
        new_tokens = []
        with open(f'../vocabs/{args.dataset}_vocab.txt', 'r') as f:
            for line in f:
                line = line[:-1]
                new_tokens.append(line)
        tokenizer.add_tokens(new_tokens)
        args.original_vocab_size = len(pretrained_tokenizer)
        args.extended_vocab_size = len(tokenizer) - args.original_vocab_size
        print(tokenizer.tokenize('co-polyimide'))
        print(pretrained_tokenizer.tokenize('co-polyimide'))
        print(len(tokenizer))
        print(len(pretrained_tokenizer))

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_class)
        args.extended_vocab_size = 0

    logger.info("\nNew merged Vocabulary size is %s" % (args.extended_vocab_size))
    
    if args.contrastive:
        train_gen, dev_gen, test_gen, label = get_batchfier(args, (pretrained_tokenizer, tokenizer))
    else:
        train_gen, dev_gen, test_gen, label = get_batchfier(args, tokenizer)

    args.n_label = len(label)

    inverse_label_map = {v: k for k, v in label.items()}
    args.tag2id = label
    args.id2tag = inverse_label_map
    if args.dataset == 'sofc_slot':
        args.id2tag[args.tag2id['B-experiment_evoking_word']] = 'O'
        args.id2tag[args.tag2id['I-experiment_evoking_word']] = 'O'

    if args.dataset == "glass_non_glass":
        model = ModelForSequenceClassification(args, args.encoder_class, n_class=args.n_label)
        evaluation_score = 'accuracy'
    else:
        model = ModelForTokenClassification(args, args.encoder_class, n_class=args.n_label)
        evaluation_score = 'macro_f1'
        
    if args.merge_version:
        new_vocab_size = len(tokenizer)
        model.cuda()
        model.resize_token_embeddings(new_num_tokens=new_vocab_size)

        # load embeddings
        print('load embeddings...')
        new_embeddings = pickle.load(open(f'../{args.dataset}.pkl', 'rb'))
        if args.dataset == "glass_non_glass":
            model.main_net.embeddings.word_embeddings.weight.data = torch.cat([model.main_net.embeddings.word_embeddings.weight.data, new_embeddings], dim=0)
        else:
            model.encoder.bert.embeddings.word_embeddings.weight.data = torch.cat([model.encoder.bert.embeddings.word_embeddings.weight.data, new_embeddings], dim=0)
        print('completed!')

    model.cuda(args.gpu)
    optimal_score = -1.0

    trainer = get_trainer(args, model, train_gen, dev_gen)
    best_dir = os.path.join(args.savename, "best_model")

    if not os.path.isdir(best_dir):
        os.makedirs(best_dir)

    results = []

    if args.do_train:
        for e in tqdm(range(0, args.n_epoch)):
            print("Epoch : {0}".format(e))
            trainer.train_epoch()
            save_path = os.path.join(args.savename, "epoch_{0}".format(e))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            if args.evaluate_during_training:
                result = trainer.test_epoch()
                if args.dataset == "glass_non_glass":
                    results.append({'accuracy': result['accuracy']})
                    print(f"Accuracy: {result['accuracy']}")
                else:
                    results.append({'macro_f1': result['macro_f1'], 'micro_f1': result['micro_f1']})
                    print(f"macro f1: {result['macro_f1']}, micro f1: {result['micro_f1']}")
                    
                if optimal_score < result[evaluation_score]:
                    optimal_score = result[evaluation_score]
                    torch.save(model.state_dict(), os.path.join(best_dir, "best_model.bin"))
                    print("Update Model checkpoints at {0}!! ".format(best_dir))

        log_full_eval_test_results_to_file(args, config=pretrained_config, results=results)


    if args.do_eval:
        result = trainer.test_epoch()
        descriptions = os.path.join(args.savename, "eval_results.txt")
        writer = open(descriptions, "w")
        if args.dataset == "glass_non_glass":
            writer.write("Accuracy: {0:.4f}".format(result['accuracy']) + "\n")
        else:
            writer.write("macro f1: {0:.4f}, micro f1 : {1:.4f}".format(result['macro_f1'], result['micro_f1']) + "\n")
        writer.close()

    if args.do_test:
        original_tokenizer = AutoTokenizer.from_pretrained(args.encoder_class)
        args.aug_word_length = len(tokenizer) - len(original_tokenizer)
        
        trainer.test_batchfier = dev_gen
        dev_results = []

        if args.model_path_list == "":
            raise EnvironmentError("require to clarify the argment of model_path")
        for model_path in args.model_path_list:
            print(model_path, "best_model", "best_model.bin")
            state_dict = torch.load(os.path.join(model_path, "best_model", "best_model.bin"))
            model.load_state_dict(state_dict)
            model.eval()

            result = trainer.test_epoch()
            dev_results.append(result[evaluation_score])

        log_full_test_results_to_file(args, test=False, config=pretrained_config, results=dev_results)
        
        trainer.test_batchfier = test_gen
        test_results = []


        if args.model_path_list == "":
            raise EnvironmentError("require to clarify the argment of model_path")
        for model_path in args.model_path_list:
            print(model_path, "best_model", "best_model.bin")
            state_dict = torch.load(os.path.join(model_path, "best_model", "best_model.bin"))
            model.load_state_dict(state_dict)
            model.eval()

            result = trainer.test_epoch()
            test_results.append(result[evaluation_score])

        log_full_test_results_to_file(args, test=True, config=pretrained_config, results=test_results)


if __name__ == "__main__":
    main()
