from transformers import AutoConfig, AutoModelForTokenClassification, AutoModel
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch
from torchcrf import CRF

model_revision = 'main'

## NER을 수행하기 위한 CRF 태거 ##
class BIO_Tag_CRF(CRF):
    def __init__(self, num_tags: int, device, batch_first: bool = False):
        super(BIO_Tag_CRF, self).__init__(num_tags=num_tags, batch_first=batch_first)
        self.device = device
        start_transitions = self.start_transitions.clone().detach()
        transitions = self.transitions.clone().detach()
        assert num_tags % 2 == 1
        num_uniq_labels = (num_tags - 1) // 2
        for i in range(num_uniq_labels, 2 * num_uniq_labels):
            start_transitions[i] = -10000
            for j in range(0, num_tags):
                if j == i or j + num_uniq_labels == i: continue
                transitions[j, i] = -10000
        self.start_transitions = nn.Parameter(start_transitions)
        self.transitions = nn.Parameter(transitions)

    def forward(self, logits, labels, masks):
        new_logits, new_labels, new_attention_mask = [], [], []
        for logit, label, mask in zip(logits, labels, masks):
            new_logits.append(logit[mask])
            new_labels.append(label[mask])
            new_attention_mask.append(torch.ones(new_labels[-1].shape[0], dtype=torch.uint8).cuda(self.device))
        
        padded_logits = pad_sequence(new_logits, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(new_labels, batch_first=True, padding_value=0)
        padded_attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

        loss = -super(BIO_Tag_CRF, self).forward(padded_logits, padded_labels, mask=padded_attention_mask, reduction='mean')
        
        if self.training:
            return (loss, )
        else:
            out = self.decode(padded_logits, mask=padded_attention_mask)
            assert(len(out) == len(labels))
            out_logits = torch.zeros_like(logits)
            for i in range(len(out)):
                k = 0
                for j in range(len(labels[i])):
                    if labels[i][j] == -100: continue
                    out_logits[i][j][out[i][k]] = 1.0
                    k += 1
                assert(k == len(out[i]))
            return (loss, out_logits, )

## NER 모델 ##
class ModelForTokenClassification(nn.Module):

    def __init__(self, args, encoder_class, n_class):
        super(ModelForTokenClassification, self).__init__()
        self.args = args
        config_kwargs = {
            'num_labels': n_class,
            'revision': model_revision,
            'use_auth_token': None,
        }
        self.config = AutoConfig.from_pretrained(encoder_class, **config_kwargs)
        self.encoder = AutoModelForTokenClassification.from_pretrained(encoder_class, from_tf=False, 
                                                                       config=self.config)
        self.classifier = BIO_Tag_CRF(n_class, args.gpu, batch_first=True)  # remove -100

    def forward(self, **inputs):
        label = inputs.pop('label')
        out = self.encoder(**inputs, output_hidden_states=True)
        logits = out['logits']
        hidden_states = out['hidden_states'][-1]
        masks = (label != -100)

        if self.args.prototype == "average":
            hidden = torch.mean(hidden_states, 1)
        elif self.args.prototype == "cls":
            hidden = hidden_states[:, 0]
        else:
            raise NotImplementedError
        if label is None:
            return None, hidden
        return self.classifier(logits, label, masks), hidden

    def resize_token_embeddings(self, new_num_tokens):

        self.encoder.bert.resize_token_embeddings(new_num_tokens)
        
    
## CLS 모델 ##
class ModelForSequenceClassification(nn.Module):

    def __init__(self, args, encoder_class, n_class):
        super(ModelForSequenceClassification, self).__init__()
        self.args = args
        self.encoder_class=encoder_class
        self.main_net = AutoModel.from_pretrained(encoder_class)
        self.hidden_size = self.main_net.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, n_class)

    def forward(self, **inputs):
        out = self.main_net(**inputs)
        hidden_states = out["last_hidden_state"]

        if self.args.prototype == "average":
            hidden = torch.mean(hidden_states, 1)
        elif self.args.prototype == "cls":
            hidden = hidden_states[:, 0]
        else:
            raise NotImplementedError
        return self.classifier(hidden_states[:, 0]), hidden


    def resize_token_embeddings(self, new_num_tokens):

        self.main_net.resize_token_embeddings(new_num_tokens)