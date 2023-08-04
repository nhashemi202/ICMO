import torch.nn as nn
import torch
from RIM_new.attention import MultiHeadAttention
from RIM_new.layer_conn_attention import LayerConnAttention
from RIM_new.BlockLSTM import BlockLSTM
import random
import time
from RIM_new.GroupLinearLayer import GroupLinearLayer
from RIM_new.sparse_grad_attn import blocked_grad

from RIM_new.blocks_core import BlocksCore

class Sparse_attention(nn.Module):
    def __init__(self, top_k = 5, num_rules=6):
        super(Sparse_attention,self).__init__()
        self.top_k = top_k
        self.num_rules = num_rules

    def forward(self, attn_s):

        # normalize the attention weights using piece-wise Linear function
        # only top k should
        attn_plot = []
        # torch.max() returns both value and location
        #attn_s_max = torch.max(attn_s, dim = 1)[0]
        #attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)
        eps = 10e-8
        time_step = attn_s.size()[2]
        bottom_k = attn_s.size()[2] - self.top_k
        delta = torch.kthvalue(attn_s, bottom_k, dim= 2)[0]
        attn_w = attn_s - delta.repeat(1, self.num_rules).unsqueeze(1)
        attn_w = torch.clamp(attn_w, min = 0)
        attn_w_sum = torch.sum(attn_w, dim = 2)
        attn_w_sum = attn_w_sum + eps 
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, self.num_rules).unsqueeze(1) 
        return attn_w_normalize


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, use_cudnn_version=True,
                 use_adaptive_softmax=False, cutoffs=None, discrete_input=True, num_blocks=6, topk=4, do_gru=False,
                 num_modules_read_input=2):
        super(RNNModel, self).__init__()
        self.topk = topk
        print('top k blocks', topk)
        self.use_cudnn_version = use_cudnn_version
        self.drop = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(0.0)
        print('number of inputs, ninp', ninp)
        if discrete_input:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Linear(ntoken, ninp)
        self.num_blocks = num_blocks
        self.nhid = nhid
        self.block_size = nhid // self.num_blocks
        print('number of blocks', self.num_blocks)
        self.discrete_input = discrete_input

        self.sigmoid = nn.Sigmoid()

        self.bc_lst = []

        print("Dropout rate", dropout)
        self.bc_lst.append(BlocksCore(nhid, 1, num_blocks, topk, True, do_gru=do_gru, num_modules_read_input=num_modules_read_input))
        self.bc_lst = nn.ModuleList(self.bc_lst)

        dropout_lst = []
        for i in range(nlayers):
            dropout_lst.append(nn.Dropout(dropout))

        print('number of layers', nlayers)
        self.dropout_lst = nn.ModuleList(dropout_lst)
        print("Make dropout lst")

        self.use_adaptive_softmax = use_adaptive_softmax
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            print('tying weights!')
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.number_of_rules = 4#num_of_rules
        self.output_ruleemb = 256
    
        self.num_gates = 2 #* self.calculate_gate_size()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        extra_loss = 0.0
        timesteps, batch_size, _ = input.shape
        emb = input
        if True:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = [[], []]
            for idx_layer in range(0, self.nlayers):
                output = []
                masklst = []
                bmasklst = []
                t0 = time.time()
                #TODO: blockify
                self.bc_lst[idx_layer].blockify_params()
                hx, cx = hidden[0][idx_layer], hidden[1][idx_layer]
                do_print = False
                for idx_step in range(input.shape[0]):
                    hx, cx, mask, bmask = self.bc_lst[idx_layer](layer_input[idx_step], hx, cx, idx_step, do_print=do_print)
                    output.append(hx)
                    masklst.append(mask)
                    bmasklst.append(bmask)

                output = torch.stack(output)
                mask = torch.stack(masklst)
                layer_input = output
                new_hidden[0].append(hx)
                new_hidden[1].append(cx)
            new_hidden[0] = torch.stack(new_hidden[0])
            new_hidden[1] = torch.stack(new_hidden[1])
            hidden = tuple(new_hidden)

        assert input.shape[1] == hx.shape[0]

        ### Step 3: Write to blocks.
        output = self.drop(output)
        dec = output.view(output.size(0) * output.size(1), self.nhid)
        dec = self.decoder(dec)
        return dec.view(output.size(0), output.size(1), dec.size(1)), hidden
    def init_hidden(self, bsz):
        weight = next(self.bc_lst[0].block_lstm.parameters())
        if True or self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
