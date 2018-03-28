import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

class BaselineModel(nn.Module):
    def __init__(self, hidden_dim, embeddings, batch_size=1, num_layers=1):
        super(BaselineModel, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        vocab_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.FloatTensor(embeddings))
        
        self.qlstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.clstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        
        self.out2end = nn.Linear(2*hidden_dim*2, 2*hidden_dim*2)
        self.out2start = nn.Linear(2*hidden_dim*2, 1)


        
    def attnCombine(self, qlstm_out, clstm_out):
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
    
    def initHidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers*direction_num, minibatch_size, hidden_dim)
        batch_size = self.batch_size
        num_layers = self.num_layers
        return (autograd.Variable(torch.zeros(num_layers*2, batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(num_layers*2, batch_size, self.hidden_dim).cuda()))

    def forward(self, question, context):
        self.qhidden = self.initHidden()
        self.chidden = self.initHidden()
        
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
        span_prob = end_prob*start_prob
        return F.normalize(span_prob.view(self.batch_size, -1), dim=1, p=1)

