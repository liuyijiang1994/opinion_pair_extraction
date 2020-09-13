import torch
from torchcrf import CRF
import common
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from module import Biaffine, GCN
from util import pool_span_representation


class NER_NET(nn.Module):
    def __init__(self, opt):
        super(NER_NET, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.bert = BertModel.from_pretrained(common.bert_path)
        self.pos_embedding = nn.Embedding(len(common.posDic),
                                          opt.pos_embed_size,
                                          padding_idx=common.posDic['PAD'])
        self.bio_label_embedding = nn.Embedding(len(common.labelDic) + 1,
                                                opt.label_embed_size,
                                                padding_idx=len(common.labelDic))

        # if opt.pos_after_gcn:
        #     gcn_input_size = self.hidden_size
        #     biaffine_input_size = self.hidden_size + opt.label_embed_size + opt.pos_embed_size
        # else:
        biaffine_input_size = self.hidden_size + opt.label_embed_size + opt.pos_embed_size
        self.gcn = GCN(in_dim=self.hidden_size,
                       mem_dim=self.hidden_size,
                       num_layers=opt.gcn_layer,
                       in_drop=opt.gcn_dropout,
                       out_drop=opt.gcn_dropout,
                       batch=True)

        self.biaffine = Biaffine(in1_features=biaffine_input_size,
                                 in2_features=biaffine_input_size,
                                 out_features=2)
        self.dropout = nn.Dropout(opt.dropout)
        self.classifier = nn.Linear(self.hidden_size + opt.pos_embed_size, len(common.labelDic))
        self.crf = CRF(len(common.labelDic), batch_first=True)

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, poses_ids, pieces_ids, pieces_mask, segment_ids, word_span, adj):
        pieces_encode, _, attention = self.bert(input_ids=pieces_ids,
                                                attention_mask=pieces_mask,
                                                token_type_ids=segment_ids,
                                                output_attentions=True)

        word_index = torch.LongTensor(range(1, common.max_pieces_length - 2)).to(poses_ids.get_device())
        word_encode = pieces_encode.index_select(dim=1, index=word_index)

        word_encode = pool_span_representation(word_encode, word_span)
        poses_encode = self.pos_embedding(poses_ids)
        word_pos_encode = torch.cat([word_encode, poses_encode], dim=-1)
        word_logits = self.classifier(word_pos_encode)

        # gcn_word_encode, gcn_mask = self.gcn(word_encode, adj)
        # gcn_word_encode = gcn_word_encode + word_encode
        # gcn_word_pos_encode = torch.cat([gcn_word_encode, poses_encode], dim=-1)

        return word_encode, word_pos_encode, word_logits

    def rel_forward(self, word_encode, gcn_word_encode, labels):
        label_encode = self.bio_label_embedding(labels)
        word_encode = torch.cat([label_encode, gcn_word_encode], dim=-1)
        rel_logits = self.biaffine(word_encode, word_encode)
        return rel_logits

    def loss(self, word_encode, gcn_word_encode, labels, word_logits, word_mask, rel_gold):
        ner_loss = -self.crf(word_logits, labels, word_mask)
        _labels = labels.clone()
        _labels[_labels == -1] = len(common.labelDic)
        rel_logits = self.rel_forward(word_encode, gcn_word_encode, _labels)
        rel_logits = rel_logits.view(-1, 2)
        rel_gold = rel_gold.view(-1)
        rel_loss = self.cross_entropy_loss(rel_logits, rel_gold)
        return ner_loss, rel_loss

    def predict(self, word_encode, gcn_word_encode, word_logits, word_mask):
        predict_labels = self.crf.decode(word_logits, word_mask)
        predict_labels_ids = pad_label_sequence(predict_labels, length=word_mask.shape[1])
        if word_mask.is_cuda:
            predict_labels_ids = predict_labels_ids.cuda(word_mask.get_device())
        rel_logits = self.rel_forward(word_encode, gcn_word_encode, predict_labels_ids)
        rel_output = F.softmax(rel_logits, dim=-1)
        return predict_labels, rel_output


def pad_label_sequence(xs, length):
    tag_ids = np.full((len(xs), length), 5)
    for idx, x in enumerate(xs):
        tag_ids[idx][:len(x)] = x

    return torch.from_numpy(tag_ids).long()
