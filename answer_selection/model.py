import torch
import torch.autograd as autograd
from torch import LongTensor, FloatTensor, ByteTensor
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import torch.optim as optim

torch.manual_seed(1)


class BaselineModel(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, vocab_size, batch_size=1, num_layers=1):
        super(BaselineModel, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.data[:2].normal_(0, 0.1)

        self.qlstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.clstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space

        self.out2end = nn.Linear(2 * hidden_dim * 2, 2 * hidden_dim * 2)
        self.out2start = nn.Linear(2 * hidden_dim * 2, 1)

    def load_embed_bioasq(self, w_dct, vec_path, type_path):
        embedding = self.word_embeddings.weight.data
        print('loading embeddings...')
        w2v_lst = defaultdict(list)
        for word_line, vec_line in zip(open(type_path, 'r', encoding='utf-8'), open(vec_path, 'r', encoding='utf-8')):
            word = word_line.strip('\n')
            if word in w_dct:
                vec = np.array([float(x) for x in vec_line.strip('\n').split()])
                w2v_lst[word].append(vec)

        for w, vec_lst in w2v_lst.items():
            embedding[w_dct[w]] = FloatTensor(np.mean(vec_lst, axis=0))

        print('load {}/{} embeddings'.format(len(w2v_lst), len(w_dct)))

    def attn_combine(self, qlstm_out, clstm_out):
        # (batch_size, hidden_size*num_directions, qseq_len)
        qlstm_out_tmp = qlstm_out.permute(1, 2, 0)
        # (batch_size, cseq_len, hidden_size*num_directions)
        clstm_out_tmp = clstm_out.permute(1, 0, 2)
        # (batch_size, cseq_len, qseq_len)
        attn_weights = F.softmax(torch.bmm(clstm_out_tmp, qlstm_out_tmp), dim=2)
        # (batch_size, qseq_len, hidden_size*num_directions)
        qlstm_out_tmp2 = qlstm_out.permute(1, 0, 2)
        # (batch_size, cseq_len, hidden_size*num_directions)
        attn_vec = torch.bmm(attn_weights, qlstm_out_tmp2)
        clstm_attn_out = torch.cat((clstm_out_tmp, attn_vec), dim=2)
        # (batch_size, cseq_len, 2*hidden_size*num_directions)
        return clstm_attn_out

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers*direction_num, minibatch_size, hidden_dim)
        batch_size = self.batch_size
        num_layers = self.num_layers
        return (autograd.Variable(torch.zeros(num_layers * 2, batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(num_layers * 2, batch_size, self.hidden_dim).cuda()))

    def forward(self, question, context):
        self.qhidden = self.init_hidden()
        self.chidden = self.init_hidden()

        qembeds = self.word_embeddings(autograd.Variable(torch.cuda.LongTensor([question])))
        cembeds = self.word_embeddings(autograd.Variable(torch.cuda.LongTensor([context])))
        # output : (seq_len, batch_size, hidden_size*num_directions)
        qlstm_out, self.qhidden = self.qlstm(
            qembeds.view(len(question), 1, -1), self.qhidden)
        clstm_out, self.chidden = self.clstm(
            cembeds.view(len(context), 1, -1), self.chidden)
        #         # (batch_size, cseq_len, 2*hidden_size*num_directions)
        #         clstm_attn_out = self.attnCombine(qlstm_out, clstm_out)

        # (batch_size, hidden_size*num_directions, qseq_len)
        qlstm_out_tmp = qlstm_out.permute(1, 2, 0)
        # (batch_size, cseq_len, hidden_size*num_directions)
        clstm_out_tmp = clstm_out.permute(1, 0, 2)
        # (batch_size, cseq_len, qseq_len)
        attn_weights = F.softmax(torch.bmm(clstm_out_tmp, qlstm_out_tmp), dim=2)
        # (batch_size, qseq_len, hidden_size*num_directions)
        qlstm_out_tmp2 = qlstm_out.permute(1, 0, 2)
        # (batch_size, cseq_len, hidden_size*num_directions)
        attn_vec = torch.bmm(attn_weights, qlstm_out_tmp2)
        clstm_attn_out = torch.cat((clstm_out_tmp, attn_vec), dim=2)
        # (batch_size, cseq_len, 2*hidden_size*num_directions)

        # (batch_size, cseq_len, 2*hidden_size*num_directions)
        end_space = self.out2end(clstm_attn_out)
        # (batch_size, 2*hidden_size*num_directions, cseq_len)
        clstm_attn_out_tmp = clstm_attn_out.permute(0, 2, 1)

        # (batch_size, cseq_len, 1)
        start_prob = F.sigmoid(self.out2start(clstm_attn_out))
        # (batch_size, cseq_len, cseq_len)
        end_prob = F.softmax(torch.bmm(end_space, clstm_attn_out_tmp), dim=2)

        # (batch_size, cseq_len, cseq_len)
        span_prob = end_prob * start_prob
        return F.normalize(span_prob.view(self.batch_size, -1), dim=1, p=1)
