import dynet as dy
import re
import numpy as np
import json
from BiRanker import BiRanker


class MaLSTMScorer(BiRanker):

    def __init__(self, m, model_file, vocab_file):
        self.LAYERS = 1
        self.INPUT_DIM = 200
        self.HIDDEN_DIM = 50
        self.vocab = {}
        self.lstm = dy.LSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, m)
        self.build_compact_model(m, model_file, vocab_file)


    def build_model(self, m, model_file, vocab_file):
        # combined.txt is current name
        print "loading vocabulary and model"
        with open(vocab_file) as f:
            vocab_size = float(f.readline().strip().split(' ')[0])
            for i, line in enumerate(f):
                fields = line.strip().split(" ")
                self.vocab[fields[0]] = i

        self.vocab["<s>"] = len(self.vocab)
        self.vocab["</s>"] = len(self.vocab)
        self.vocab["UNK"] = len(self.vocab)
        print "vocabulary loaded"

        self.lookup = m.add_lookup_parameters((vocab_size + 3, self.INPUT_DIM))
        # model1 is current name
        self.lookup, self.lstm = dy.load(model_file, m)
        print "model file loaded"

    def build_compact_model(self, m, model_file, vocab_file):
        # combined.txt is current name
        print "loading vocabulary and model"
        with open(vocab_file) as f:
            vocab_size = float(f.readline().strip().split(' ')[0])
            for i, line in enumerate(f):
                fields = line.strip().split("\t")
                self.vocab[fields[0]] = int(fields[1])

        print "vocabulary loaded"

        self.lookup = m.add_lookup_parameters((vocab_size, self.INPUT_DIM))
        # model1 is current name
        self.lookup, self.lstm = dy.load(model_file, m)
        print "model file loaded"

    def tokenize_sentence(self, doc_in):
        toks = []
        arr = doc_in.split(". ")
        for sent in arr:
            if sent.strip() == '':
                continue
            sent_toks = ["<s>"]
            sent_toks.extend(self.bioclean(sent))
            sent_toks.append("</s>")
            toks.extend(sent_toks)

        return toks


    def process_input(self, file_path, train=True):
        data = []
        with open(file_path, "r") as infile:
            for line in infile:
                entry = []
                if train:
                    (q, doc, score) = line.split('\t')
                else:
                    (q, doc) = line.split('\t')
                entry.append(self.tokenize_sentence(q))
                entry.append(self.tokenize_sentence(doc))
                if train:
                    entry.append(float(score))
                data.append(entry)

        return data


    def bioclean(self, t):
        return re.sub('[.,?;*!%^&_+():-\\[\]{}]', '', t.replace('"', '').replace('/', '').
        replace('\\', '').replace("'", '').strip().lower()).split()


    def do_one_sentence_pair(self, sent1, sent2):
        dy.renew_cg()
        s0 = self.lstm.initial_state()

        query = [self.lookup[self.vocab[tok]] if tok in self.vocab else self.lookup[self.vocab["UNK"]] for tok in sent1[:75]]
        doc = [self.lookup[self.vocab[tok]] if tok in self.vocab else self.lookup[self.vocab["UNK"]] for tok in sent2[:75]]

        output1 = s0.transduce(query)
        output2 = s0.transduce(doc)
        diff = dy.l1_distance(output1[-1], output2[-1])
        return dy.exp(-1 * dy.l2_norm(diff)).npvalue()
        #return 1 - (dy.exp(-1 * dy.l2_norm(diff)).npvalue())

    def single_comp(self, question, sentence):
        q_toks = self.tokenize_sentence(question)
        snip_toks = self.tokenize_sentence(sentence)
        pred = self.do_one_sentence_pair(q_toks, snip_toks)

        return pred

    def getRankedList(self, question):
        if 'snippets' not in question or len(question['snippets']) == 0:
            print "no snippets to rank for question:", question['body']
            return []

        candidates = []
        q_toks = self.tokenize_sentence(question['body'])
        sentences = list(set(self.getSentences(question)))
        for sentence in sentences:
            snip_toks = self.tokenize_sentence(sentence)
            pred = self.do_one_sentence_pair(q_toks, snip_toks)
            candidates.append((sentence, pred))

        list.sort(candidates, key=lambda cand: cand[1], reverse=True)
        return [x[0] for x in candidates[:10]]


def main():
    with open("/home/ubuntu/BioASQ-trainingDataset4b.json", "r") as infile:
        data = json.load(infile)
    print data['questions'][0]['body']
    m = dy.ParameterCollection()
    malstm = MaLSTMScorer(m, "/home/ubuntu/model_dropout_corrected20", "/home/ubuntu/model_vocab.txt")
    #test_data = malstm.process_input("frac_small.txt")
    snippets = malstm.getRankedList(data['questions'][0])
    print data['questions'][0]['body']
    for item in snippets:
        print item

    '''for sentence in test_data:
        pred = malstm.do_one_sentence_pair(sentence[0], sentence[1])
        #print "prediction:", 1 - pred.npvalue()
        print "prediction:", pred'''


if __name__ == '__main__':
    main()
