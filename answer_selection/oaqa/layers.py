import torch
import torch.autograd as autograd
from torch import LongTensor, FloatTensor, ByteTensor
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np


class CharEmbed(nn.Module):
    def __init__(self, vocab_size, window_size=5, embed_dim=50, output_dim=200):
        super(CharEmbed, self).__init__()
        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.char_embed.weight.data.normal_(0, 0.1)
        self.conv = nn.Conv2d(1, output_dim, (window_size, embed_dim), padding=((window_size - 1)//2, 0))

    # input: batch_size*char_num
    def forward(self, char_sequences):
        # batch_size*word_len*embed_dim
        char_embedding = self.char_embed(char_sequences)
        # batch_size*c_out*conv_word_len [*conv_embed_dim(1)]
        conv_result = self.conv(char_embedding.unsqueeze(1)).squeeze(3)
        # batch_size*c_out
        word_char_embed = F.max_pool1d(conv_result, conv_result.size(2)).squeeze(2)
        # batch_size*1*c_out
        return word_char_embed.unsqueeze(1)
