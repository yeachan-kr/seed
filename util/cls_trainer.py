from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
import torch
import torch.nn as nn
from model.losses import NTXentLoss,AlignLoss
from model.classification_model import ModelForSequenceClassification
from overrides import overrides
from sklearn.metrics import accuracy_score
from itertools import chain

## Classification을 수행하는 Classification Trainer ##
class CLSTrainer:
    def __init__(self, args, model: ModelForSequenceClassification, train_batchfier, test_batchfier, optimizers, scheduler,
                 update_step, clip_norm):
        self.args = args
        self.model = model
        self.train_batchfier = train_batchfier
        self.test_batchfier = test_batchfier
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.criteria = nn.CrossEntropyLoss(ignore_index=-100)
        self.step = 0
        self.update_step = update_step
        self.clip_norm = clip_norm

    def reformat_inp(self, inp):
        raise NotImplementedError

    def train_epoch(self):
        return NotImplementedError

    def test_epoch(self):
        return NotImplementedError

## Contrastive Learning 을 적용하지 않는 Classification Trainer ##
class CFCLSTrainer(CLSTrainer):
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers, scheduler,
                 update_step, clip_norm, n_label):
        super(CFCLSTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers, scheduler,
                                        update_step, clip_norm)
        self.n_label = n_label

    @overrides
    def reformat_inp(self, inp):
        inp_tensor = {i: inp[i].to("cuda") for i in inp}
        return inp_tensor

    def train_epoch(self):

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers
        scheduler = self.scheduler

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar = 0, 0, 0, 0
        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        for inputs in pbar:
            inputs = self.reformat_inp(inputs)
            label = inputs.pop('label')

            logits, _ = model(**inputs)
            loss = criteria(logits.view(-1,logits.size(-1)), label.view(-1))

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

        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.eval()
        model.zero_grad()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        tot_score = 0.0

        true_buff = []
        eval_buff = []

        for inputs in pbar:
            with torch.no_grad():
                inputs = self.reformat_inp(inputs)
                label = inputs.pop('label').view(-1)
                logits, _ = model(**inputs)
                preds = torch.argmax(logits, -1)
                preds = preds.view(-1)
                loss = criteria(logits.view(-1, logits.size(-1)), label)
                step_loss += loss.item()
                pbar_cnt += 1

            true_buff.append(label.tolist())
            eval_buff.append(preds.tolist())
            score = torch.mean((preds == label).to(torch.float))
            tot_score += score

            pbar.set_description(
                "test loss : %f  test accuracy : %f" % (
                    step_loss / pbar_cnt, tot_score / pbar_cnt), )
            pbar.update()
        pbar.close()
        true_buff= list(chain(*true_buff))
        eval_buff = list(chain(*eval_buff))
        accuracy = accuracy_score(true_buff,eval_buff)

        return {'accuracy': accuracy}


## Contrastive Learning 을 적용하는 Trainer ##
class ContrastiveCLSTrainer(CLSTrainer):
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers, scheduler,
                 update_step, clip_norm, n_label):
        super(ContrastiveCLSTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers, scheduler,
                                                 update_step, clip_norm)
        self.n_label = n_label
        if args.align_type=="cosine":
            self.align_loss = AlignLoss(args, args.per_gpu_train_batch_size, temperature=1.0)
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

        return inp_tensor, domain_tensor, label

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
                                   collate_fn=batchfier.collate, pin_memory=True,drop_last=True)

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar = 0, 0, 0, 0

        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        for inp in pbar:
            inp, domain_inp, label = self.reformat_inp(inp)

            _, hi = model(**inp)  # Bert Tokens
            logits, hj = model(**domain_inp) # Domain Tokens

            loss_align = self.align_loss(hi, hj) ## Contrastive Loss
            loss_ce = self.criteria(logits.view(-1,logits.size(-1)), label.view(-1))

            loss = loss_align + loss_ce
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

        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.eval()
        model.zero_grad()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        tot_score = 0.0

        true_buff = []
        eval_buff = []

        for inp in pbar:
            with torch.no_grad():
                inp, domain_inp, label = self.reformat_inp(inp) 
                logits, _ = model(**domain_inp) ## test시에는 domain token들만 사용
                preds = torch.argmax(logits, -1)
                preds = preds.view(-1)
                label = label.view(-1)
                loss = criteria(logits.view(-1,logits.size(-1)), label)
                step_loss += loss.item()
                pbar_cnt += 1

            true_buff.append(label.tolist())
            eval_buff.append(preds.tolist())

            score = torch.mean((preds == label).to(torch.float))
            tot_score += score

            pbar.set_description(
                "test loss : %f  test accuracy : %f" % (
                    step_loss / pbar_cnt, tot_score / pbar_cnt), )
            pbar.update()
        pbar.close()

        true_buff= list(chain(*true_buff))
        eval_buff = list(chain(*eval_buff))
        accuracy = accuracy_score(true_buff,eval_buff)
        
        return {'accuracy': accuracy}


