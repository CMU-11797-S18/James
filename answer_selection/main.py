from __future__ import unicode_literals
import sys
import torch
import dill
import pickle
import json
import time
import pandas as pd
import numpy as np
import random
from io import open
import re
from utils import get_qa_pair, add_span, get_word_dict, bioclean, text_to_list, squad_to_bioasq_format, load_embedding
from utils import evaluate, flatten_span_list, formalize_data, get_char_dict, Dictionary

from model import BaselineModel
from sklearn.model_selection import train_test_split

def original_script():
    with open('./data/BioASQ-trainingDataset5b.txt', encoding='utf-8') as f:
        data = json.load(f)

    print('getting qa pair from data')
    qa_pair = get_qa_pair(data)

    print('adding spans into the qa_pair')
    qa_pair = add_span(qa_pair)
    qa_pair = text_to_list(qa_pair)

    bioword_dict = get_word_dict(qa_pair, bioclean)
    char_dict = get_char_dict(qa_pair)

    train, test = train_test_split(qa_pair, test_size=0.2, random_state=32)
    train, validate = train_test_split(train, test_size=0.2, random_state=32)
    train_flat = formalize_data(flatten_span_list(train), bioword_dict, char_dict)
    test_flat = formalize_data(flatten_span_list(test), bioword_dict, char_dict)
    validate_flat = formalize_data(flatten_span_list(validate), bioword_dict, char_dict)


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_train_data(path, bioword_dict=None, gloveword_dict=None, char_dict=None):
    bioword_dict = Dictionary(clean_func=bioclean) if bioword_dict is None else bioword_dict
    char_dict = Dictionary() if char_dict is None else char_dict
    gloveword_dict = Dictionary(clean_func=unicode.lower) if gloveword_dict is None else gloveword_dict
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    print('getting qa pair from data')
    qa_pair = get_qa_pair(data)

    print('adding spans into the qa_pair')
    qa_pair = add_span(qa_pair)
    qa_pair = text_to_list(qa_pair)

    bioword_dict = get_word_dict(qa_pair, dct=bioword_dict)
    gloveword_dict = get_word_dict(qa_pair, dct=gloveword_dict)
    char_dict = get_char_dict(qa_pair, dct=char_dict)

    return qa_pair, bioword_dict, char_dict, gloveword_dict

if __name__ == '__main__':
    set_seed(1)
    mode = sys.argv[1]
    if mode == 'bioasq':
        train_path = '../oaqa/input/5b_train.json'
        test_path = '../oaqa/input/5b_test.json'
    else:
        train_path = '../oaqa/input/msmarco_test_train.json'
        test_path = '../oaqa/input/msmarco_test_test.json'
    train, bioword_dict, char_dict, gloveword_dict = get_train_data(train_path)
    test, bioword_dict, char_dict, gloveword_dict = get_train_data(test_path, bioword_dict, gloveword_dict, char_dict)
    bioasq_test, bioword_dict, char_dict, gloveword_dict = get_train_data('../oaqa/input/5b_test.json', bioword_dict, gloveword_dict, char_dict)

    train, validate = train_test_split(train, test_size=0.2, random_state=32)

    if mode == 'bioasq':
        target_dict = bioword_dict
        word_embed_dim = 200
    else:
        target_dict = gloveword_dict
        word_embed_dim = 300

    train_flat = formalize_data(flatten_span_list(train), target_dict, char_dict)
    validate_flat = formalize_data(flatten_span_list(validate), target_dict, char_dict)
    test_flat = formalize_data(flatten_span_list(test), target_dict, char_dict)
    bioasq_test_flat = formalize_data(flatten_span_list(bioasq_test), target_dict, char_dict)

    epoch_num = 10
    hidden_dim = 100
    model = BaselineModel(hidden_dim, word_embed_dim, len(target_dict), len(char_dict))
    model.cuda()

    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters())

    if mode == 'bioasq':
        model.load_embed_bioasq(target_dict, 'vectors.txt', 'types.txt')
    else:
        w2embed = load_embedding(target_dict, '../data/glove.6B.{}d.txt'.format(word_embed_dim))
        for w, embedding in w2embed.items():
            model.word_embeddings.weight.data[gloveword_dict[w]].copy_(torch.from_numpy(embedding.astype('float32')))

    evaluate(validate_flat, model, 'validation', target_dict, char_dict)
    start = time.time()

    best_acc = 0

    for epoch in range(epoch_num):
        print('epoch:', epoch)
        model.train()
        random.shuffle(train_flat)

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
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)

            optimizer.step()

        print('train loss:', np.mean(train_loss))
        print('train acc:', np.mean(train_acc))

        #evaluate(train_flat, model, 'train', bioword_dict, char_dict)
        dev_acc = evaluate(validate_flat, model, 'validation', target_dict, char_dict)
        evaluate(test_flat, model, 'test', target_dict, char_dict)
        evaluate(bioasq_test_flat, model, 'bioasq test', target_dict, char_dict)
        print('time elapsed:', time.time() - start)

        if epoch >= 5 and dev_acc > best_acc:
            print ('saving models...')
            best_acc = dev_acc
            torch.save(model, './model')

            with open('word_dict.pkl', 'wb') as output:
                pickle.dump(target_dict, output)

            with open('char_dict.pkl', 'wb') as output:
                pickle.dump(char_dict, output)
        print ()




