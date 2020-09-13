import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1

        # linear: dm1 -> dm2 x out
        affine = self.linear(input1)  # batch, len1, out_features x dm2
        affine = affine.view(batch_size, len1 * self.out_features, dim2)  # batch, len1 x out_features, dm2

        input2 = torch.transpose(input2, 1, 2)  # batch_size, dim2, len2

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)
        # batch, len1 x out, len2 -> batch, len2, len1 x out

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        #  batch, len2, len1 x out  -> batch, len2, len1, out

        biaffine = torch.transpose(biaffine, 1, 2).contiguous()
        # batch, len1, len2, out

        return biaffine


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, in_dim, mem_dim, num_layers, in_drop=0.5, out_drop=0.5, batch=False):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = in_dim

        self.in_drop = nn.Dropout(in_drop)
        self.gcn_drop = nn.Dropout(out_drop)

        # gcn layer
        self.W = nn.ModuleList()
        self.batch = batch
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, token_encode, adj):
        '''
        :param adj:  batch, seqlen, seqlen
        :param token_encode: batch, seqlen, dm
        :return:
        '''

        if not self.batch:
            adj = adj.unsqueeze(0)
            token_encode = token_encode.unsqueeze(0)

        embs = self.in_drop(token_encode)

        gcn_inputs = embs

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return gcn_inputs, mask
