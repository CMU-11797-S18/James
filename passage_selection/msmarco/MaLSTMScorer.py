import dynet as dy
import re
import numpy as np
import json
from BiRanker import BiRanker


class MaLSTMScorer(BiRanker):

    def __init__(self, m, model_file, vocab_file):
        self.LAYERS = 1
        self.INPUT_DIM = 300
        self.HIDDEN_DIM = 50
        self.vocab = {}
        self.lstm = dy.LSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, m)
        self.build_model(m, model_file, vocab_file)
        self.pattern = re.compile('[\W_]+')


    def build_model(self, m, model_file, vocab_file):
        print "loading vocabulary and model"
        with open(vocab_file) as f:
            self.vocab = json.load(f)
        print "vocabulary loaded"

        self.lookup = m.add_lookup_parameters((len(self.vocab), self.INPUT_DIM))
        self.lookup, self.lstm = dy.load(model_file, m)
        print "model file loaded"


    def tokenize_sentence(self, doc_in):
        toks = []
        arr = doc_in.split(". ")
        for sent in arr:
            if sent.strip() == '':
                continue
            sent_toks = self.get_tokens(sent)
            sent_toks.append("</s>")
            toks.extend(sent_toks)

        return toks


    def get_tokens(self, string_in):
        no_unicode = string_in.encode("ascii", errors="ignore").decode()
        return self.pattern.sub(' ', no_unicode).strip().lower().split()


    def do_one_sentence_pair(self, sent1, sent2):
        dy.renew_cg()
        s0 = self.lstm.initial_state()

        query = [self.lookup[self.vocab[tok]] if tok in self.vocab else self.lookup[self.vocab["UNK"]] for tok in sent1[:75]]
        doc = [self.lookup[self.vocab[tok]] if tok in self.vocab else self.lookup[self.vocab["UNK"]] for tok in sent2[:75]]

        output1 = s0.transduce(query)
        output2 = s0.transduce(doc)
        diff = dy.l1_distance(output1[-1], output2[-1])
        return dy.exp(-1 * dy.l2_norm(diff)).npvalue()


    def single_comp(self, question, sentence):
        q_toks = self.tokenize_sentence(question)
        snip_toks = self.tokenize_sentence(sentence)
        pred = self.do_one_sentence_pair(q_toks, snip_toks)

        return pred


    def getRankedList(self, question, dev_eval=False):
        if not question["snippets"]:
            print "no snippets to rank for question:", question['body']
            return []

        candidates = []
        q_toks = self.tokenize_sentence(question['body'])
        #sentences = list(set(self.getSentences(question)))
        for doc in question["snippets"]:
            snip_toks = self.tokenize_sentence(doc["text"])
            pred = self.do_one_sentence_pair(q_toks, snip_toks)
            if dev_eval:
                candidates.append((doc["text"], pred, doc["is_selected"])) 
            else:
                candidates.append((doc["text"], pred))

        list.sort(candidates, key=lambda cand: cand[1], reverse=True)
        if dev_eval:
            return candidates[:10]
        else:
            return [x[0] for x in candidates[:10]]


def read_data(target_file):
    item_dict = {}
    with open(target_file, "r") as infile:
        item_dict = json.load(infile)

    return item_dict['questions']


def main():
    print "reading dev data..."
    data = read_data("msmarco_test.json")
    m = dy.ParameterCollection()
    malstm = MaLSTMScorer(m, "models/msmarco_model_nodrop2", "model_vocab_complete.json")
    correct_count1 = 0
    correct_count3 = 0
    print "evaluating accuracy on", len(data), "queries..."
    for ind, data_point in enumerate(data):
        if ind % 1000 == 0 and ind > 0:
            print "on query", ind + 1

        snippets = malstm.getRankedList(data_point, dev_eval=True)
        for x, item in enumerate(snippets):
            if item[2] == 1:
                if x == 0:
                    correct_count1 += 1
                    correct_count3 += 1
                    break
                elif x < 3:
                    correct_count3 += 1
                    break
                else:
                    break

    # there are 2622 queries with no correct snippets, so correct for them here manually
    print "accuracy for top snippet:", 1.0 * correct_count1 / (len(data) - 2622)
    print "accuracy for top 3 snippets:", 1.0 * correct_count3 / (len(data) - 2622)


if __name__ == '__main__':
    main()
