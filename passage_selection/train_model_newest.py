import re
import dynet as dy
import numpy as np
import random
import math
import sys


def tokenize_sentence(doc_in):
    toks = []
    arr = doc_in.split(". ")
    for sent in arr:
        if sent.strip() == '':
            continue
        sent_toks = []
        sent_toks.extend(bioclean(sent))
        sent_toks.append("</s>")
        toks.extend(sent_toks)

    return toks


def process_input(file_path, train=True):
    data = []
    with open(file_path, "r") as infile:
        for line in infile:
            entry = []
            if train:
                (q, doc, score) = line.split('\t')
            else:
                (q, doc) = line.split('\t')
            entry.append(tokenize_sentence(q))
            entry.append(tokenize_sentence(doc))
            if train:
                entry.append(float(score))
            data.append(entry)

    return data


bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\\[\]{}]', '', t.replace('"', '').replace('/', '').
                            replace('\\', '').replace("'", '').strip().lower()).split()

train_data = process_input("training_data_new.txt")
test_data = process_input("frac_test_clean.txt")
data_ind = range(len(train_data))

LAYERS = 1
INPUT_DIM = 200
HIDDEN_DIM = 50

print "initalizing model..."
vocab = {}
used_vocab = {}
model = dy.ParameterCollection()
vocab_count = 0

with open("used_vocab.txt") as f:
    for line in f:
        line = line.strip()
        used_vocab[line] = True

with open("combined.txt") as f:
    vocab_size = float(f.readline().strip().split(' ')[0])
    lookup = model.add_lookup_parameters((len(used_vocab) + 2, INPUT_DIM))
    for i, line in enumerate(f):
        fields = line.strip().split(" ")
        if fields[0].strip() in used_vocab:
            lookup.init_row(vocab_count, list(map(float, fields[1:])))
            vocab[fields[0]] = vocab_count
            vocab_count += 1

bound = math.sqrt(6.0 / (vocab_size + INPUT_DIM))
vocab["</s>"] = vocab_count + 1
lookup.init_row(vocab["</s>"], [random.uniform(-bound, bound) for x in xrange(INPUT_DIM)])
vocab["UNK"] = vocab_count + 2
lookup.init_row(vocab["UNK"], [random.uniform(-bound, bound) for x in xrange(INPUT_DIM)])
print "loaded vocab for", vocab_count, "words"

with open("model_vocab.txt", "w") as outfile:
    outfile.write(str(len(used_vocab) + 2) + '\n')
    for key in vocab:
        outfile.write(key + '\t' + str(vocab[key]) + '\n')

trainer = dy.AdamTrainer(model)
lstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
lstm.set_dropout(0.3)


def do_one_sentence_pair(lstm, sent1, sent2):
    s0 = lstm.initial_state()

    query = [lookup[vocab[tok]] if tok in vocab else lookup[vocab["UNK"]] for tok in sent1[:75]]
    doc = [lookup[vocab[tok]] if tok in vocab else lookup[vocab["UNK"]] for tok in sent2[:75]]

    output1 = s0.transduce(query)
    output2 = s0.transduce(doc)
    diff = dy.l1_distance(output1[-1], output2[-1])
    return dy.exp(-1 * dy.l2_norm(diff))
    #diff = dy.cdiv(dy.dot_product(output1[-1], output2[-1]), (dy.l2_norm(output1[-1]) * dy.l2_norm(output2[-1])))
    #return dy.exp(-1 * dy.abs(diff))


print "starting training..."
sum_of_losses = 0.0
for epoch in range(10):
    dy.renew_cg()
    losses = []
    random.shuffle(data_ind)
    total_loss = 0
    print "on epoch", epoch
    for x, ind in enumerate(data_ind):
        pred = do_one_sentence_pair(lstm, train_data[ind][0], train_data[ind][1])
        loss = dy.squared_distance(pred, dy.scalarInput(train_data[ind][2]))
        #loss = dy.squared_distance(pred, dy.scalarInput(1 - train_data[ind][2]))
        losses.append(loss)
        if x % 5000 == 0:
            print "on sentence", x

        if x % 64 == 0 and x > 0 or x == len(data_ind) - 1:
            batch_loss = dy.esum(losses) / len(losses)
            total_loss += batch_loss.npvalue()
            batch_loss.backward()
            trainer.update()
            losses = []
            dy.renew_cg()

    test_loss = 0.0
    for x, instance in enumerate(test_data):
        pred = do_one_sentence_pair(lstm, instance[0], instance[1])
        loss = dy.squared_distance(pred, dy.scalarInput(instance[2]))
        #loss = dy.squared_distance(pred, dy.scalarInput(1 - instance[2]))
        losses.append(loss)

        if x % 64 == 0 and x > 0 or x == len(test_data) - 1:
            test_loss += dy.esum(losses).npvalue() / len(losses)
            losses = []
            dy.renew_cg()

    print "total loss for epoch ", epoch, ":", total_loss
    print "total test loss for epoch", epoch, ":", test_loss

dy.save("model_dropout_corrected" + str(epoch+1), [lookup, lstm])
