
from RIM_new.rnn_models import RNNModel #rnn_models
import torch
import torch.nn as nn


class BlocksWrapper(nn.Module):
    def __init__(self, ntokens, nhid, nout, dropout=0.0, num_blocks=4, update_topk=2, num_blocks_read_input=2):
        super(BlocksWrapper, self).__init__()
        self.myrnn = RNNModel("GRU", ntokens, nhid, nhid,
                            nlayers=1, dropout=dropout, tie_weights=False,
                            use_cudnn_version=False, use_adaptive_softmax=False,
                            cutoffs=[10000], discrete_input=False, num_blocks=num_blocks,
                            topk=update_topk, do_gru=False, num_modules_read_input=num_blocks_read_input)
        #self.myrnn = nn.LSTM(ntokens, nhid)
        self.nhid = nhid

        print('using blocks wrapper!')

    def forward(self, inp, h):
        h[0] = h[0].unsqueeze(0)
        h[1] = h[1].unsqueeze(0)
        inp = inp.unsqueeze(0)

        hx = h[0].contiguous()
        cx = h[1].contiguous()
        ob, (hx,cx) = self.myrnn(inp, (hx, cx))
        hx = hx.squeeze(0)
        cx = cx.squeeze(0)
        return (hx,cx)


if __name__ == "__main__":
    nhid = 128
    ntokens = 128

    blocks = BlocksWrapper(ntokens, nhid, n_out=nhid).cuda()
    gru = torch.nn.GRU(ntokens, nhid).cuda()

    x = torch.randn(1, 1, ntokens).cuda()

    h0 = torch.randn(1, 1, nhid).cuda()
    h0_blocks = torch.randn(1, 1, nhid*2).cuda()

    og, hg = gru(x, h0)
    print('gru of x: o,h', og.shape, hg.shape)

    ob, hb = blocks(x, h0_blocks)
    print('block res: o,h', ob.shape, hb.shape)



