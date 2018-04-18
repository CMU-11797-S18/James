import pickle
import torch
import dill
from utils import get_chars_ind_lst, word_tokenize, get_utf8
import numpy as np

class SpanSelector():
    def __init__(self, model_path='/home/ubuntu/model', word_dict_path='/home/ubuntu/word_dict.pkl',
                 char_dict_path='/home/ubuntu/char_dict.pkl'):
        self.model = torch.load(model_path)
        self.word_dict = pickle.load(open(word_dict_path))
        self.char_dict = pickle.load(open(char_dict_path))


    def predict(self, question, snippets):
        model = self.model
        word_dict = self.word_dict
        char_dict = self.char_dict

        best_answer = None
        best_answer_prob = None

        q_words = [word_dict[w] for w in word_tokenize(question)]
        q_chars = get_chars_ind_lst(char_dict, word_tokenize(question))
        for snippet in snippets:
            d_words = [word_dict[w] for w in word_tokenize(snippet)]
            d_chars = get_chars_ind_lst(char_dict, word_tokenize(snippet))
            pred = model(q_words, d_words, q_chars, d_chars)
            pred_prob = np.max(pred[0].data.cpu().numpy())
            pred_index = np.argmax(pred[0].data.cpu())
            start, end = pred_index//len(d_words), pred_index%len(d_words)
            if best_answer_prob is None or pred_prob > best_answer_prob:
                best_answer_prob = pred_prob
                best_answer = ' '.join(word_tokenize(snippet)[start:end + 1])
        if best_answer is None:
            print ('-----NULL ANSWER')
            return '', 0

        return best_answer, best_answer_prob
