from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
import torch
from model.losses import NTXentLoss,AlignLoss
from overrides import overrides
from itertools import chain
import numpy as np
from . import conlleval


## NER을 수행하는 Trainer ##
class NERTrainer:
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers, scheduler,
                 update_step, clip_norm, n_label):
        self.args = args
        self.model = model
        self.train_batchfier = train_batchfier
        self.test_batchfier = test_batchfier
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.step = 0
        self.update_step = update_step
        self.clip_norm = clip_norm
        self.n_label = n_label

        self.id2tag = args.id2tag
        self.tag2id = args.tag2id
        cnt = dict()
        for label in args.tag2id:
            if label[0] in ['I', 'B']: tag = label[2:]
            else: continue
            if tag not in cnt: cnt[tag] = 1
            else: cnt[tag] += 1

        self.eval_labels = sorted([l for l in cnt.keys() if l != 'experiment_evoking_word'])

    def reformat_inp(self, inp):
        raise NotImplementedError

    def train_epoch(self):
        return NotImplementedError

    def test_epoch(self):
        return NotImplementedError

    ## Macro F1, Micro F1 score를 계산하는 함수 ##
    def compute_metrics(self, labels, predictions):
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [self.id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        preds, labs = [], []
        for pred, lab in zip(true_predictions, true_labels):
            preds.extend(pred)
            labs.extend(lab)
        assert(len(preds) == len(labs))
        labels_and_predictions = [" ".join([str(i), labs[i], preds[i]]) for i in range(len(labs))]
        counts = conlleval.evaluate(labels_and_predictions)
        scores = conlleval.get_scores(counts)
        results = {}
        macro_f1 = 0
        for k in self.eval_labels:
            if k in scores:
                results[k] = scores[k][-1]
            else:
                results[k] = 0.0
            macro_f1 += results[k]
        macro_f1 /= len(self.eval_labels)
        results['macro_f1'] = macro_f1 / 100
        results['micro_f1'] = conlleval.metrics(counts)[0].fscore
        return results

## Contrastive Learning을 적용하지 않는 NERTrainer ##
class CFNERTrainer(NERTrainer):
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers, scheduler,
                 update_step, clip_norm, n_label):
        super(CFNERTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers, scheduler,
                                        update_step, clip_norm, n_label)
        

    @overrides
    def reformat_inp(self, inp):
        inp_tensor = {i: inp[i].to("cuda") for i in inp}
        return inp_tensor

    def train_epoch(self):
        model = self.model
        batchfier = self.train_batchfier
        optimizer = self.optimizers
        scheduler = self.scheduler

        if isinstance(batchfier, IterableDataset):
            print("IterableDatset")            
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=batchfier.collate_ner, pin_memory=True)
        model.train()
        tot_loss, step_loss, tot_cnt, n_bar = 0, 0, 0, 0
        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        for inputs in pbar:
            inputs = self.reformat_inp(inputs)
            outputs, _ = model(**inputs)
            loss = outputs[0]
        
            # new_tokens_mask = (inputs['input_ids'] > 31090).float()
            # if new_tokens_mask.sum() > 0:
            #     # labels = inputs['input_ids']
            #     # inputs['input_ids'][inputs['input_ids'] > 31090] == 104 # for [mask] tokens
        
            #     labels = torch.ones_like(inputs['input_ids']) * -1
            #     labels[inputs['input_ids'] > 31090] = inputs['input_ids'][inputs['input_ids'] > 31090]
            #     inputs['input_ids'][inputs['input_ids'] > 31090] == 104 # for [mask] tokens
                
            #     outputs, _ = model(**inputs)
            

            loss = loss / self.update_step
            step_loss += loss.item()
            tot_loss += loss.item()
            tot_cnt += 1

            loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                scheduler.step(self.step)
                model.zero_grad()
                pbar.set_description(
                    "training loss : %f  , iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt),
                         n_bar), )
                pbar.update()

        pbar.close()

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier


        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate_ner, pin_memory=True)

        model.eval()
        model.zero_grad()
        pbar = tqdm(batchfier)

        true_buff = []
        eval_buff = []

        for inputs in pbar:
            with torch.no_grad():
                inputs = self.reformat_inp(inputs)
                outputs, _ = model(**inputs)

            true_buff.append(inputs['label'].tolist())
            eval_buff.append(outputs[1].tolist())

        true_buff= list(chain(*true_buff))
        eval_buff = list(chain(*eval_buff))

        results = self.compute_metrics(true_buff, eval_buff)
        
        return results



## Contrastive Learning을 적용하지 않는 NER Trainer ##
class ContrastiveNERTrainer(NERTrainer):
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers, scheduler,
                 update_step, clip_norm, n_label):
        super(ContrastiveNERTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers, scheduler, 
                                                 update_step, clip_norm, n_label)
        if args.align_type=="cosine":
            self.align_loss = AlignLoss(args, args.per_gpu_train_batch_size, temperature=args.temperature)
        elif args.align_type=="simclr":
            self.align_loss = NTXentLoss(args, args.per_gpu_train_batch_size,temperature=args.temperature)
        else:
            print(args.align_type)
            raise NotImplementedError
        
    @overrides
    def reformat_inp(self, inp):
        inp_tensor = dict()
        domain_tensor = dict()
        label = None
        for key in inp.keys():
            if key == 'label':
                label = inp[key].to('cuda')
            elif "domain" in key:
                domain_tensor[key.split("_", 1)[1]] = inp[key].to("cuda")
            else:
                inp_tensor[key] = inp[key].to('cuda')
        inp_tensor['label'] = None
        domain_tensor['label'] = label
        return inp_tensor, domain_tensor

    def train_epoch(self):

        model = self.model
        batchfier = self.train_batchfier
        optimizer = self.optimizers
        scheduler = self.scheduler


        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=batchfier.collate_ner, pin_memory=True,drop_last=True)

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar = 0, 0, 0, 0

        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)
        
        for inputs in pbar:
            inp_tensor, domain_tensor = self.reformat_inp(inputs)
            _, hi = model(**inp_tensor)  # Bert Tokens
            outputs, hj = model(**domain_tensor) # Domain Tokens

            loss = outputs[0]
            loss_align = self.align_loss(hi, hj)

            loss += loss_align
            
            loss /= self.update_step
            
            step_loss += loss.item()
            tot_loss += loss.item()
            tot_cnt += 1

            loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                scheduler.step(self.step)
                model.zero_grad()
                pbar.set_description(
                    "training loss : %f , iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt),
                         n_bar), )
                pbar.update()


        pbar.close()

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier


        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate_ner, pin_memory=True)

        model.eval()
        model.zero_grad()
        pbar = tqdm(batchfier)

        true_buff = []
        eval_buff = []

        for inputs in pbar:
            with torch.no_grad():
                _, domain_tensor = self.reformat_inp(inputs)
                outputs, _ = model(**domain_tensor)

            true_buff.append(domain_tensor['label'].tolist())
            eval_buff.append(outputs[1].tolist())

        true_buff= list(chain(*true_buff))
        eval_buff = list(chain(*eval_buff))

        results = self.compute_metrics(true_buff, eval_buff)

        return results