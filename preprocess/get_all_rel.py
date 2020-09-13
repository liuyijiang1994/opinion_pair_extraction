import torch
import common
import re
import stanza
import numpy as np
from transformers import *
import operator

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained('F:\\resources\pretrained_model\\bert-base-uncased', do_lower_case=True)
split_token = {'?', '.', '!'}
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True)
rel_set = set()
pos_set = set()


def read_data(data_list, data_type):
    seq = 0
    datasets = []
    words = []
    labels = []
    relations = []
    relations_gold = []
    with open(f'./data/{data_list}.{data_type}', 'r') as f:
        for l in f:
            if l.strip() == "#Relations":
                continue
            elif l.strip() == "" and len(words) > 0:
                # if "B-T" in labels or "B-P" in labels:
                datasets.append(
                    {"words": words, "labels": labels, "relations": relations, "relations_gold": relations_gold})
                if len(words) > seq:
                    seq = len(words)
                words = []
                labels = []
                relations = []
                relations_gold = []
            elif len(l.strip().split("\t")) == 2:
                tempLine = l.strip().split("\t")
                # WORD
                words.append(tempLine[0].lower())
                # LABEL
                labels.append(tempLine[1])
            elif len(l.strip().split("\t")) == 4:
                rel = list(map(int, l.strip().split("\t")))
                relations_gold.append([rel[2], rel[3], rel[0], rel[1]])
                if -1 not in rel:
                    relations.append(rel)

    return datasets


def flat_deptree(doc, length):
    adj = np.zeros((length, length))
    currnet_len = 0
    for sent in doc.sentences:
        for word in sent.words:
            rel_set.add(word.deprel)
            pos_set.add(word.pos)


def split_sentences(words):
    sentence_list = []
    sentece = []
    for word in words:
        sentece.append(word)
        if word in split_token:
            sentence_list.append(sentece)
            sentece = []
    if len(sentece) != 0:
        sentence_list.append(sentece)

    return sentence_list


def parse_sentence_lsit(sentence_list, word_len):
    doc = nlp(sentence_list)
    adj = flat_deptree(doc, word_len)
    return adj


def convert_dataset(datasets):
    final_data_list = []
    for data in datasets:
        words = data['words']
        pieces = []
        word_span = []
        labels = data['labels']
        labels_id = [common.labelDic[label] for label in labels]
        relations = data['relations']
        relation = np.zeros((len(words), len(words)))
        for r in relations:
            relation[r[1] - 1][r[3] - 1] = 1
            relation[r[3] - 1][r[1] - 1] = 1
        current_idx = 0
        pieces.append("[CLS]")
        for word in words:
            t_piece = tokenizer.tokenize(word)
            pieces.extend(t_piece)
            word_span.append([current_idx, current_idx + len(t_piece) - 1])
            current_idx += len(t_piece)
        sentence_list = split_sentences(words)
        assert len(word_span) == len(words) == len(labels_id)
        assert len(pieces) - 1 == word_span[-1][-1] + 1
        pieces.append("[SEP]")
        adj = parse_sentence_lsit(sentence_list, len(words))

    return final_data_list


for data_list in ['2014LapStandard', '2014ResStandard', '2015ResStandard']:
    data_set = {}
    for data_type in ['train', 'test']:
        dataset = read_data(data_list, data_type)
        convert_dataset(dataset)
print(rel_set)
print(pos_set)
