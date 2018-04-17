from __future__ import unicode_literals
import torch
import pickle
import json
import time
import pandas as pd
import numpy as np
import random
import nltk
from io import open
import re
from utils import get_qa_pair, add_span, get_word_dict, bioclean, text_to_list, squad_to_bioasq_format
from utils import evaluate, flatten_span_list, formalize_data, get_char_dict

from model import BaselineModel
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    with open('./data/BioASQ-trainingDataset5b.txt', encoding='utf-8') as f:
        data = json.load(f)

    print('getting qa pair from data')
    qa_pair = get_qa_pair(data)

    print('adding spans into the qa_pair')
    qa_pair = add_span(qa_pair)
    qa_pair = text_to_list(qa_pair)
    # # Debugging span extraction
    # for (snippet, spans) in zip(qa_pair[0][3], qa_pair[0][4]):
    #     print (spans)
    #     print (snippet)
    #     for (start, end) in spans:
    #         print (snippet[start:end + 1])

    bioword_dict = get_word_dict(qa_pair, bioclean)
    char_dict = get_char_dict(qa_pair)

    train, test = train_test_split(qa_pair, test_size=0.2, random_state=32)
    train, validate = train_test_split(train, test_size=0.2, random_state=32)
    train_flat = formalize_data(flatten_span_list(train), bioword_dict, char_dict)
    test_flat = formalize_data(flatten_span_list(test), bioword_dict, char_dict)
    validate_flat = formalize_data(flatten_span_list(validate), bioword_dict, char_dict)


    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    epoch_num = 50
    hidden_dim = 100
    model = BaselineModel(hidden_dim, 200, len(bioword_dict), len(char_dict))
    model.cuda()
    model.load_embed_bioasq(bioword_dict, './data/vectors.txt', './data/types.txt')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    evaluate(validate_flat, model, 'validation', bioword_dict, char_dict)
    start = time.time()

    for epoch in range(epoch_num):
        print('epoch:', epoch)
        model.train()
        # random.shuffle(train_flat)

        train_acc = []
        train_loss = []
        for sample in train_flat:
            span = sample.span
            index = len(sample.d_words) * span[0] + span[1]

            optimizer.zero_grad()
            pred = model(sample.q_words, sample.d_words, sample.q_chars, sample.d_chars)
            loss = -torch.log(pred.index_select(1, torch.autograd.Variable(torch.cuda.LongTensor([index]), requires_grad=False)))
            loss.backward()

            pred_index = np.argmax(pred[0].data.cpu())
            train_acc.append(pred_index == index)
            train_loss.append(loss.data.cpu()[0])
            optimizer.step()

        print('train loss:', np.mean(train_loss))
        print('train acc:', np.mean(train_acc))

        evaluate(train_flat, model, 'train', bioword_dict, char_dict)
        evaluate(validate_flat, model, 'validation', bioword_dict, char_dict)
        evaluate(test_flat, model, 'test', bioword_dict, char_dict)
        print('time elapsed:', time.time() - start)

        if epoch == 13:
            torch.save(model, './model')

            with open('word_dict.pkl', 'wb') as output:
                pickle.dump(bioword_dict, output)

            with open('char_dict.pkl', 'wb') as output:
                pickle.dump(char_dict, output)

            break


