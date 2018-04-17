from __future__ import unicode_literals
from io import open
import torch
import torch.autograd as autograd
from torch import LongTensor, FloatTensor, ByteTensor
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from layers import CharEmbed
import torch.optim as optim

torch.manual_seed(1)


class BaselineModel(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, vocab_size, char_vocab_size, window_size=5, char_embed_dim=50,
                 word_char_embed_dim=200, batch_size=1, num_layers=1, use_char_embed=True):
        super(BaselineModel, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_char_embed = use_char_embed

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.data[:2].normal_(0, 0.1)

        self.char_embed = CharEmbed(char_vocab_size, window_size=window_size, embed_dim=char_embed_dim, output_dim=word_char_embed_dim)

        embed_size = embedding_dim
        if use_char_embed:
            embed_size += word_char_embed_dim

        self.qlstm = nn.LSTM(embed_size, hidden_dim, bidirectional=True)
        self.clstm = nn.LSTM(embed_size, hidden_dim, bidirectional=True)

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
        return (Variable(torch.zeros(num_layers*2, batch_size, self.hidden_dim).cuda()),
                Variable(torch.zeros(num_layers*2, batch_size, self.hidden_dim).cuda()))

    def forward(self, question, context, question_chars, context_chars):
        question = Variable(torch.cuda.LongTensor([question]))
        context = Variable(torch.cuda.LongTensor([context]))
        question_chars = [question_chars]
        context_chars = [context_chars]

        # question_chars should be batch_size*seq_len*word_len
        self.qhidden = self.init_hidden()
        self.chidden = self.init_hidden()

        # batch_size*seq_len*embed_dim
        q_word_embeds = self.word_embeddings(question)
        c_word_embeds = self.word_embeddings(context)

        q_char_embed_lst = list()
        for batch_idx in range(len(question_chars)):
            batch_data = []
            for i in range(len(question_chars[batch_idx])):
                batch_data.append(self.char_embed(Variable(torch.cuda.LongTensor([question_chars[batch_idx][i]]))))
            q_char_embed_lst.append(torch.cat(batch_data, dim=1))
        q_char_embeds = torch.cat(q_char_embed_lst, dim=0)

        c_char_embed_lst = list()
        for batch_idx in range(len(context_chars)):
            batch_data = []
            for i in range(len(context_chars[batch_idx])):
                batch_data.append(self.char_embed(Variable(torch.cuda.LongTensor([context_chars[batch_idx][i]]))))
            c_char_embed_lst.append(torch.cat(batch_data, dim=1))
        c_char_embeds = torch.cat(c_char_embed_lst, dim=0)

        if self.use_char_embed:
            qembeds = torch.cat([q_word_embeds, q_char_embeds], dim=2)
            cembeds = torch.cat([c_word_embeds, c_char_embeds], dim=2)
        else:
            qembeds = q_word_embeds
            cembeds = c_word_embeds

        # output : (seq_len, batch_size, hidden_size*num_directions)
        qlstm_out, self.qhidden = self.qlstm(
            qembeds.permute(1, 0, 2), self.qhidden)
        clstm_out, self.chidden = self.clstm(
            cembeds.permute(1, 0, 2), self.chidden)

        # (batch_size, cseq_len, 2*hidden_size*num_directions)
        clstm_attn_out = self.attn_combine(qlstm_out, clstm_out)

        # (batch_size, cseq_len, 2*hidden_size*num_directions)
        end_space = self.out2end(clstm_attn_out)
        # (batch_size, 2*hidden_size*num_directions, cseq_len)
        clstm_attn_out_tmp = clstm_attn_out.permute(0, 2, 1)

        # (batch_size, cseq_len, 1)
        start_prob = F.sigmoid(self.out2start(clstm_attn_out))
        # (batch_size, cseq_len, cseq_len)
        end_weight = torch.bmm(end_space, clstm_attn_out_tmp)
        end_constraint = torch.cuda.FloatTensor(end_weight.size()).fill_(0)
        for i in range(1, end_weight.size(1)):
            end_constraint[:, i, :i].fill_(-float('inf'))
        end_weight += Variable(end_constraint, requires_grad=False)

        end_prob = F.softmax(end_weight, dim=2)

        # (batch_size, cseq_len, cseq_len)
        span_prob = end_prob * start_prob
        return F.normalize(span_prob.view(self.batch_size, -1), dim=1, p=1)
