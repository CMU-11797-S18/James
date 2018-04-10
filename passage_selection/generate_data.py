## generates positive training data
## instead of treating all snippets equally, outputs a fractional score
## based on the proportion of answer tokens in each snippet

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
import numpy as np
import glob
import re
import sys

# english stopwords from nltk. copied here so nltk doesn't need to be loaded
stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'use', 'used']
feats = []
vectorizer = TfidfVectorizer()
#line_count = 0

# set up the tf-idf scores for extracting tokens
def setup_tfidf():
    docs = []
    with open("all_sorted.txt", "r") as infile:
        for line in infile:
            docs.append(re.sub('[.,?;*!%^&_+():\[\]{}"]', '', line).strip())

    vectorizer.fit(docs)
    feats.extend(vectorizer.get_feature_names())


def process_file(outfile, file_list):
    # hold out questions from phaseB_4b_04.json as test data
    test_data = {}
    with open("old/phaseB_4b_04.json", "r") as infile:
        data = json.load(infile)
        for q in data['questions']:
            test_data[q['body'].strip().replace('\n', ' ').encode("utf-8")] = True

    # process question-snippet pairs from all json files in current directory
    for infile in file_list:
        with open(infile, "r") as f:
            d = json.load(f)
            q_count = 0
            for item in d['questions']:
                qn = item['body'].strip().replace('\n', ' ').encode("utf-8")
                # skip questions in the test data
                if qn not in test_data:
                    # create a list of tokens from high tf-idf terms in the answer
                    # process each question type differently because answer fields have different structures
                    if 'snippets' in item:
                        ans_toks = []
                        if item['type'] == 'factoid':
                            ans = item['exact_answer'][0]
                            if type(ans) == list:
                                ans = ans[0]
                            ans_toks = top_tokens(ans)

                        elif item['type'] == 'list':
                            if type(item['exact_answer']) == list:
                                ans = list(chain.from_iterable(item['exact_answer']))
                            else:
                                ans = item['exact_answer']
                            ans_toks = []
                            for answer in ans:
                                toks = re.sub('[.,?;*!%^&_+():\[\]{}"]', '', answer).lower().strip().split()
                                for tok in toks:
                                    if tok not in stopwords and len(tok) > 2:
                                        ans_toks.append(tok)

                        # summary type only has ideal_answer field
                        elif item['type'] == 'summary':
                            if type(item['ideal_answer']) == list:
                                ans = item['ideal_answer'][0].split('.')[0]
                            else:
                                ans = item['ideal_answer'].split('.')[0]
                            ans_toks = top_tokens(ans)


                        """elif item['type'] == 'yesno':
                            ans = qn
                            ans_toks = top_tokens(item['body'])"""


                        # add tokens from ideal_answer field
                        if item['type'] != 'summary' and 'ideal_answer' in item:
                            if type(item['ideal_answer']) == list:
                                for ideal in item['ideal_answer']:
                                    ans_toks.extend(top_tokens(ideal))
                            else:
                                ans_toks.extend(top_tokens(item['ideal_answer']))

                        ans_toks.extend(top_tokens(item['body']))

                        if len(ans_toks) < 1:
                            print "**error: no tokens found for answer " + str(ans)
                            print "question # ", q_count

                        # get tokens from sentence in each snippet and calculate score
                        for snippet in item['snippets']:
                            snip_sentences = snippet['text'].split(". ")
                            for cand_sentence in snip_sentences:
                                sn_toks = sn_tokens(cand_sentence)
                                if len(sn_toks) == 0:
                                #    print "**error: no tokens found for snippet " + snippet['text']
                                #    print "question # ", q_count
                                    continue

                                match_count = 0
                                for tok in sn_toks:
                                    if tok in ans_toks:
                                        match_count += 1

                                score = 1.0 * match_count / len(sn_toks)
                                sn = cand_sentence.strip().encode("utf-8")
                                outfile.write(qn + '\t' + sn + '\t' + str(score) + '\n')
                                """print score, qn
                                print ans, ans_toks
                                print snippet['text'], sn_toks"""

                q_count += 1
                #if q_count >= 2:
                #    break


def top_tokens(sentence, max_tok=100):
    s_mod = re.sub('[.,?;*!%^&_+():\[\]{}"]', '', sentence).strip()
    #print s_mod
    v = vectorizer.transform([s_mod])
    feature_index = v.nonzero()[1]
    scores = zip(feature_index, [v[0, x] for x in feature_index])
    scores.sort(key=lambda x:x[1], reverse=True)
    ret_list = []
    for item in scores:
        #if item[1] < 0.15 or len(ret_list) >= max_tok:
        #    break
        if (feats[item[0]] not in stopwords and len(feats[item[0]]) > 2 and item[1] < 0.15) or feats[item[0]].isnumeric():
            ret_list.append(feats[item[0]])

    if len(ret_list) == 0:
        for item in scores:
            if feats[item[0]] not in stopwords:
                ret_list.append(feats[item[0]])

    return ret_list

def sn_tokens(sentence):
    s_mod = re.sub('[.,?;*!%^&_+():\[\]{}"]', '', sentence).strip().lower().split()
    ret_list = [word for word in s_mod if word not in stopwords]
    return ret_list


if __name__ == "__main__":
    files = glob.glob("./*.json")
    #files = ["old/phaseB_4b_04.json"]
    if len(files) == 0:
        print "no files to process"
        sys.exit(1)

    setup_tfidf()
    with open("frac_test.txt", "w") as outfile:
        process_file(outfile, files)
