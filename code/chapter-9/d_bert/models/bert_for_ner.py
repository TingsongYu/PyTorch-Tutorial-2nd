import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from transformers import BertModel, BertPreTrainedModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy


class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores


class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config, ):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs
