import unicodedata
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
import re


class Dictionary:
    def __init__(self):
        dct = dict()
        # NULL for padding, UNK for unknown word
        dct['<NULL>'] = 0
        dct['<UNK>'] = 1
        self.dct = dct

    def __contains__(self, item):
        item = Dictionary.normalize(item)
        return item in self.dct

    def __setitem__(self, key, value):
        key = Dictionary.normalize(key)
        self.dct[key] = value

    def __getitem__(self, item):
        item = Dictionary.normalize(item)
        return self.dct.get(item, 1)

    def add(self, key):
        key = Dictionary.normalize(key)
        if key not in self.dct:
            self.dct[key] = len(self.dct)

    @staticmethod
    def normalize(w):
        return unicodedata.normalize('NFD', w)


def get_qa_pair(data, q_types=['factoid']):
    """
    :param data: data having the same format as BioASQ data
    :param q_types: desired question types
    :return: qa_pair having the question, answer list, question type, snippet
    """

    # Unify answer format and separate punctuations by spaces.
    def unify_answer_format(obj, a):
        if obj['type'] == 'factoid' and not isinstance(a[0], list):
            a = [a]
        unified_a = []
        for answer_lst in a:
            unified_answer_lst = [separate_punctuation_by_space(synonym) for synonym in answer_lst]
            unified_a.append(unified_answer_lst)
        return unified_a

    # Separate snippets, de-duplicate them and separate punctuations by spaces.
    def get_snippets(obj):
        dup_count = 0
        snippets = []
        for snippet in obj['snippets']:
            snippet = snippet['text']
            if snippet not in s:
                snippets.append(snippet)
            else:
                dup_count += 1
        return snippets, dup_count

    qa_pair = []
    duplicate_snippet_count = 0
    for obj in data['questions']:
        if obj['type'] in q_types:
            q = separate_punctuation_by_space(obj['body'])
            a = unify_answer_format(obj, obj['exact_answer'])
            t = obj['type']
            s, count = get_snippets(obj)
            duplicate_snippet_count += count

            if len(s) > 0:
                qa_pair.append([q, a, t, s])
    print('duplicate snippet count:', duplicate_snippet_count)
    print('questions count:', len(qa_pair))
    return qa_pair


def separate_punctuation_by_space(text):
    return ' '.join(word_tokenize(text))


def get_spans(a, snippet):
    snippet = snippet.lower()
    spans = []
    for synonym_lst in a:
        for synonym in synonym_lst:
            synonym = synonym.lower()
            synonym = re.escape(synonym)

            # Get the start position in the text
            span = [(m.start(), m.end() - 1) for m in re.finditer(synonym, snippet)]

            # Convert them into positions in the word list
            span = [(len(snippet[:start].split(' ')) - 1, len(snippet[:end + 1].split(' ')) - 1) for (start, end) in
                    span]
            if len(span):
                spans += span
    return spans


def add_span(qa_pair):
    new_qa_pair = []
    for (q, a, t, s) in qa_pair:
        new_s, new_span_lst = [], []
        span_lst = [get_spans(a, snippet) for snippet in s]
        for (snippet, spans) in zip(s, span_lst):
            if len(spans):
                new_s.append(snippet)
                new_span_lst.append(spans)
        new_qa_pair.append([q, a, t, new_s, new_span_lst])
    return new_qa_pair


def bioclean(t):
    return ' '.join(re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                           (t.replace('"', '').replace('/', '')
                            .replace('\\', '').replace("'", '')
                            .strip().lower()))
                    .split())


def get_word_dict(qa_pair, clean_func, dct=None):
    dct = Dictionary() if dct is None else dct
    for (q, a, t, s, span_lst) in qa_pair:
        for word in q:
            dct.add(clean_func(word))
        for ans_lst in a:
            for ans in ans_lst:
                for word in ans:
                    dct.add(clean_func(word))
        for snippet in s:
            for word in snippet:
                dct.add(clean_func(word))
    return dct


def load_embed_bioasq(w_dct, vec_path, type_path, embedding):
    print('loading embeddings...')
    w2v_lst = defaultdict(list)
    for word_line, vec_line in zip(vec_path, type_path):
        word = word_line.strip('\n')
        if word in w_dct:
            vec = np.array(vec_line.strip('\n').split())
            w2v_lst[word].append(vec)

    for w, vec_lst in w2v_lst.items():
        embedding[w_dct[w]] = np.mean(vec_lst, axis=0)

    print('load {}/{} embeddings'.format(len(w2v_lst), len(w_dct)))


def flatten_span_list(qa_pair):
    new_qa_pair = []
    for (q, s, span_lst, t) in qa_pair:
        for (snippet, spans) in zip(s, span_lst):
            for span in spans:
                new_qa_pair.append([q, snippet, span, t])
    return new_qa_pair
