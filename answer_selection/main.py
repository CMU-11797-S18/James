import torch
import json
import pandas as pd
import numpy as np
import random
import nltk
import re
from collections import defaultdict
from utils import get_qa_pair, add_span, get_word_dict, bioclean, text_to_list, squad_to_bioasq_format, flatten_span_list, formalize_data
from model import BaselineModel
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    with open('./data/BioASQ-trainingDataset5b.txt', encoding='utf8') as f:
        data = json.load(f)

    print('getting qa pair from data')
    qa_pair = get_qa_pair(data)
    # qa_pair = squad_to_bioasq_format(squad_data)

    print('adding spans into the qa_pair')
    qa_pair = add_span(qa_pair)
    # (q, a, t, s, span_lst)
    qa_pair = text_to_list(qa_pair)
    # # Debugging span extraction
    # for (snippet, spans) in zip(qa_pair[0][3], qa_pair[0][4]):
    #     print (spans)
    #     print (snippet)
    #     for (start, end) in spans:
    #         print (snippet[start:end + 1])

    bioword_dict = get_word_dict(qa_pair, bioclean)
    train, test = train_test_split(qa_pair, test_size=0.2, random_state=32)
    train, validate = train_test_split(train, test_size=0.2, random_state=32)
    train_flat = formalize_data(flatten_span_list(train), bioword_dict)
    test_flat = formalize_data(flatten_span_list(test), bioword_dict)
    validate_flat = formalize_data(flatten_span_list(validate), bioword_dict)

    hidden_dim = 100
    model = BaselineModel(hidden_dim, 200, len(bioword_dict))
    model.load_embed_bioasq(bioword_dict, './data/vectors.txt', './data/types.txt')

