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
    train, test = train_test_split(qa_pair, test_size=0.2, random_state=32)
    train, validate = train_test_split(train, test_size=0.2, random_state=32)
    train_flat = formalize_data(flatten_span_list(train), bioword_dict)
    test_flat = formalize_data(flatten_span_list(test), bioword_dict)
    validate_flat = formalize_data(flatten_span_list(validate), bioword_dict)

    import time

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    epoch_num = 50
    hidden_dim = 100
    model = BaselineModel(hidden_dim, 200, len(bioword_dict))
    model.cuda()
    model.load_embed_bioasq(bioword_dict, './data/vectors.txt', './data/types.txt')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    dev_acc = []
    dev_loss = []
    for sample in squad_validate_flat:
        q = sample.q_words.numpy().tolist()
        context = sample.d_words.numpy().tolist()
        span = sample.span
        index = len(context) * span[0] + span[1]
        pred = model(q, context)

        loss = -torch.log(pred.index_select(1, torch.autograd.Variable(torch.cuda.LongTensor([index]))))

        pred_index = np.argmax(pred[0].data.cpu())
        dev_acc.append(pred_index == index)
        dev_loss.append(loss.data.cpu()[0])
    print('validation acc:', np.mean(dev_acc))
    print('validation loss:', np.mean(dev_loss))

    for epoch in range(epoch_num):
        print('epoch:', epoch)
        model.train()
        # random.shuffle(squad_train_flat)

        train_acc = []
        train_loss = []
        for sample in squad_train_flat:
            q = sample.q_words.numpy().tolist()
            context = sample.d_words.numpy().tolist()
            span = sample.span
            index = len(context) * span[0] + span[1]

            optimizer.zero_grad()
            pred = model(q, context)
            loss = -torch.log(pred.index_select(1, torch.autograd.Variable(torch.cuda.LongTensor([index]))))
            loss.backward()

            pred_index = np.argmax(pred[0].data.cpu())
            train_acc.append(pred_index == index)
            train_loss.append(loss.data.cpu()[0])
            optimizer.step()

        print('train loss:', np.mean(train_loss))
        print('train acc:', np.mean(train_acc))

        model.eval()
        dev_acc = []
        dev_loss = []
        for sample in squad_validate_flat:
            q = sample.q_words.numpy().tolist()
            context = sample.d_words.numpy().tolist()
            span = sample.span
            index = len(context) * span[0] + span[1]
            pred = model(q, context)

            loss = -torch.log(pred.index_select(1, torch.autograd.Variable(torch.cuda.LongTensor([index]))))

            pred_index = np.argmax(pred[0].data.cpu())

            pred_span = (pred_index // len(context), pred_index % len(context))

            dev_acc.append(pred_index == index)
            dev_loss.append(loss.data.cpu()[0])

        print('validation acc:', np.mean(dev_acc))
        print('validation loss:', np.mean(dev_loss))

        model.eval()
        test_acc = []
        test_loss = []
        for sample in squad_test_flat:
            q = sample.q_words.numpy().tolist()
            context = sample.d_words.numpy().tolist()
            span = sample.span
            index = len(context) * span[0] + span[1]
            pred = model(q, context)

            loss = -torch.log(pred.index_select(1, torch.autograd.Variable(torch.cuda.LongTensor([index]))))

            pred_index = np.argmax(pred[0].data.cpu())
            test_acc.append(pred_index == index)
            test_loss.append(loss.data.cpu()[0])
        print('test acc:', np.mean(test_acc))
        print('test loss:', np.mean(test_loss))

