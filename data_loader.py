import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
import common
import numpy as np


class InputFeature(object):
    def __init__(self, words, word_span, word_mask, poses_ids, labels_id, pieces, pieces_ids, pieces_mask, segment_ids,
                 relations, gold_relations, adj):
        self.words = words
        self.word_span = word_span
        self.word_mask = word_mask
        self.poses_ids = poses_ids
        self.labels_id = labels_id
        self.pieces = pieces
        self.pieces_ids = pieces_ids
        self.pieces_mask = pieces_mask
        self.segment_ids = segment_ids
        self.relations = relations
        self.gold_relations = gold_relations
        self.adj = adj


def create_batch_iter(mode, opt):
    dataset = torch.load(common.dataset)
    dataset = dataset[mode]
    features = convert_examples_to_features(dataset)
    poses_ids = torch.tensor([f.poses_ids for f in features], dtype=torch.long)
    labels_id = torch.tensor([f.labels_id for f in features], dtype=torch.long)
    pieces_ids = torch.tensor([f.pieces_ids for f in features], dtype=torch.long)
    pieces_mask = torch.tensor([f.pieces_mask for f in features], dtype=torch.uint8)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    relations = torch.tensor([f.relations for f in features], dtype=torch.long)
    adj = torch.tensor([f.adj for f in features], dtype=torch.float)
    word_span = torch.tensor([f.word_span for f in features], dtype=torch.long)
    word_mask = torch.tensor([f.word_mask for f in features], dtype=torch.uint8)
    words = [f.words for f in features]
    words = [' '.join(word) for word in words]
    pieces = [f.pieces for f in features]
    pieces = [' '.join(piece) for piece in pieces]
    # print('pieces_ids', pieces_ids.shape)
    tensor_dataset = MyTensorDataset(words, word_span, word_mask, poses_ids, labels_id, pieces, pieces_ids, pieces_mask,
                                     segment_ids, relations, adj)
    batch_size = opt.train_batch_size
    if mode == "train":
        sampler = RandomSampler(tensor_dataset)
    elif mode == "dev" or mode == 'test':
        sampler = SequentialSampler(tensor_dataset)
        batch_size = opt.test_batch_size
    iterator = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=2, pin_memory=True)
    len_dataset = len(dataset)
    return iterator, len_dataset


class MyTensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, words, word_span, word_mask, poses_ids, labels_id, pieces, pieces_ids, pieces_mask, segment_ids,
                 relations,
                 adj):
        tensors = [word_span, word_mask, poses_ids, labels_id, pieces_ids, pieces_mask, segment_ids, relations, adj]
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.text_list = words
        self.pieces_list = pieces

    def __getitem__(self, index):
        return self.text_list[index], self.pieces_list[index], tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


def convert_examples_to_features(data_list):
    features = []
    cont = 0
    r_cont = 0
    for iddata, data in enumerate(data_list):
        word_span = data['word_span'] + [[-1, -1]] * (common.max_word_length - len(data['words']))
        word_mask = [1] * len(data['word_span'])
        word_mask += [0] * (common.max_word_length - len(data['word_span']))
        segment_ids = [0] * common.max_pieces_length
        pieces_mask = [1] * len(data['pieces_ids']) + [0] * (common.max_pieces_length - len(data['pieces_ids']))
        pieces_ids = data['pieces_ids'] + [0] * (common.max_pieces_length - len(data['pieces_ids']))
        labels_id = data['labels_id'] + [-1] * (common.max_word_length - len(data['words']))
        poses_ids = data['poses_ids'] + [common.posDic['PAD']] * (common.max_word_length - len(data['words']))
        relations = np.full((common.max_word_length, common.max_word_length), -1)
        relations[:len(data['words']), :len(data['words'])] = 0
        # print(data['words'])
        if len(data['relations']) == 0:
            cont += 1
        r_cont += len(data['relations'])
        for r in data['relations']:
            for idx in range(r[0], r[1]):
                for idy in range(r[2], r[3]):
                    relations[idx][idy] = 1
            for idx in range(r[2], r[3]):
                for idy in range(r[0], r[1]):
                    relations[idx][idy] = 1
            # print(data['words'][r[0]:r[1]], data['words'][r[2]:r[3]])
        # print()
        adj = np.zeros((common.max_word_length, common.max_word_length))
        for d in data['deprels']:
            adj[d[0]][d[1]] = 1
            adj[d[1]][d[0]] = 1
        feature = InputFeature(words=data['words'],
                               word_span=word_span,
                               word_mask=word_mask,
                               poses_ids=poses_ids,
                               labels_id=labels_id,
                               pieces=data['pieces'],
                               pieces_ids=pieces_ids,
                               pieces_mask=pieces_mask,
                               segment_ids=segment_ids,
                               relations=relations,
                               gold_relations=data['relations'],
                               adj=adj)
        features.append(feature)
    print(cont, r_cont, len(features))
    return features


if __name__ == '__main__':
    training_iter, num_train_steps = create_batch_iter('train')
    for step, batch in enumerate(training_iter):
        words, pieces, batch = batch
        word_span, word_mask, poses_ids, labels_id, pieces_ids, pieces_mask, segment_ids, relations, adj = batch
        word_span = word_span.numpy().tolist()
        for p, ws in zip(pieces, word_span):
            p = p.split()
            p = p[1:-1]
            for t in ws:
                if -1 not in t:
                    print(p[t[0]:t[1] + 1])
