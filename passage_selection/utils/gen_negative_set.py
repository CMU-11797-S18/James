import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import random

stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
questions = {}
vectorizer = TfidfVectorizer()
feats = []
MAX_LINES = int(1000000*3/8)

def setup_tfidf():
    docs = []
    with open("all_sorted.txt", "r") as infile:
        for line in infile:
            docs.append(re.sub('[.,?;*!%^&_+():\[\]{}"]', '', line).strip())

    vectorizer.fit(docs)
    feats.extend(vectorizer.get_feature_names())


def process_file(outfile, file_list):
    line = 0
    for infile in file_list:
        with open(infile, "r") as f:
            d = json.load(f)
            for item in d['questions']:
                if(item['body'] in questions):
                    continue

                questions[item['body']] = True
                line += 1
                ran_doc_list = []
                out_docs = []
                out_rand = []
                qn = item['body'].strip().replace('\n', ' ').encode("utf-8")
                q_toks = top_tokens(qn, 5)

                target_docs = 0
                if 'snippets' in item:
                    target_docs = 4 * len(item['snippets'])
                else:
                    target_docs = 4

                ran_doc_list = random.sample(xrange(MAX_LINES), int(target_docs/2))

                # answer is fectoid type
                if item['type'] == 'factoid':
                    ans = item['exact_answer'][0]
                    if type(ans) == list:
                        ans = ans[0]
                    ans_toks = top_tokens(ans)
                    out_docs, out_rand = get_snips(q_toks, ans_toks, target_docs, ran_doc_list)

                # answer is list type
                elif item['type'] == 'list':
                    ans_toks = []

                    # iterate through exact answer, which is list of list of strings
                    for ans_list in item['exact_answer']:
                        for inner_elem in ans_list:
                            # get rid of punctuation and split into tokens
                            single_ans = re.sub('[.,?;*!%^&_+():\[\]{}"]', '', inner_elem).strip().split()
                            # non-stopword tokens added to list of answer tokens
                            for token in single_ans:
                                if token not in stopwords and len(token) > 2:
                                    ans_toks.append(token)

                    out_docs, out_rand = get_snips(q_toks, ans_toks, target_docs, ran_doc_list)

                # answer is summary type
                elif item['type'] == 'summary':
                    ans_toks = []
                    if type(item['ideal_answer']) == list:
                        ans_sent = item['ideal_answer'][0].split('.')
                    else:
                        ans_sent = item['ideal_answer'].split('.')

                    # take first two sentences
                    if len(ans_sent) > 2:
                        ans_sent = ans_sent[0:2]

                    for sent in ans_sent:
                        if len(ans_sent) > 1:
                            ans_toks.extend(top_tokens(sent, 5))
                        else:
                            ans_toks.extend(top_tokens(sent))

                    out_docs, out_rand = get_snips(q_toks, ans_toks, target_docs, ran_doc_list)


                # answer is yesno type
                elif item['type'] == 'yesno':
                    if 'snippets' not in item or not item['snippets']:
                        continue

                    ans_toks = top_tokens(item['snippets'][0]['text'])
                    if len(item['snippets'][0]) > 1:
                        ans_toks = ans_toks[:5]
                        ans_toks.extend(top_tokens(item['snippets'][0]['text'], 5))

                    out_docs, out_rand = get_snips(q_toks, ans_toks, target_docs, ran_doc_list)

                for doc in out_docs:
                    doc = doc.replace('\n', ' ').strip().encode("utf-8")
                    outfile.write(qn + '\t' + doc + '\t' + '0' + '\n')
                for doc in out_rand:
                    doc = doc.replace('\n', ' ').strip().encode("utf-8")
                    outfile.write(qn + '\t' + doc + '\t' + '0' + '\n')
                    


def top_tokens(sentence, max_tok=10):
    s_mod = re.sub('[.,?;*!%^&_+():\[\]{}"]', '', sentence).strip()
    #print s_mod
    v = vectorizer.transform([s_mod])
    feature_index = v.nonzero()[1]
    scores = zip(feature_index, [v[0, x] for x in feature_index])
    scores.sort(key=lambda x:x[1], reverse=True)
    ret_list = []
    for item in scores:
        if item[1] < 0.1 or len(ret_list) >= max_tok:
            break
        if feats[item[0]] not in stopwords and len(feats[item[0]]) > 2:
            ret_list.append(feats[item[0]])

    return ret_list


def get_snips(candidate_toks, a_toks, num, doc_list):
    ## go through each doc, filter by a_toks, then pick any with 1 or more q_tok matches
    ## at the same time take docs that match the line numbers in doc_list
    ## return list of snippets
    q_toks = [tok for tok in candidate_toks if tok not in a_toks]
    if not q_toks:
        q_toks.append(candidate_toks[0])

    #print q_toks
    #print a_toks
    line_num = 0
    rand_docs = []
    match_1 = []
    match_2 = []
    match_3 = []
    max_line = 0
    if doc_list:
        max_line = max(doc_list)
    with open("./fixed_out08/data_small.txt", "r") as infile:
        for line in infile:
            doc = unicode(line, "utf-8")
            #doc = line.split('\t')[2]
            ll = doc.lower()
            q_match = 0
            a_match = 0

            for tok in q_toks:
                if tok in ll:
                    q_match += 1
            for tok in a_toks:
                if tok in ll:
                    a_match += 1
            if a_match == 0:
                if q_match == 1:
                    match_1.append(doc)
                elif q_match == 2:
                    match_2.append(doc)
                elif q_match >= 3:
                    match_3.append(doc)
            if (match_1 + match_2 + match_3) > num * 4 and (len(rand_docs) >= num or line_num > max_line):
                break

            if line_num in doc_list and q_match == 0 and a_match == 0:
                rand_docs.append(doc)

            line_num += 1

    ret_list = []
    if len(match_3) <= num/4:
        ret_list.extend(match_3)
    else:
        ret_list.extend(get_rand_docs(match_3, num/4))

    if len(match_2) <= num/4:
        ret_list.extend(match_2)
    else:
        ret_list.extend(get_rand_docs(match_2, num/4))

    if int(0.75 * num) - len(ret_list) >= len(match_1):
        ret_list.extend(match_1)
    else:
        ret_list.extend(get_rand_docs(match_1, int(0.75 * num) - len(ret_list)))

    print len(match_1), len(match_2), len(match_3), len(ret_list), len(rand_docs), num
    return (ret_list, rand_docs[:num/4])


def get_rand_docs(doc_list, num):
    ret_list = []
    targets = random.sample(xrange(len(doc_list)), num)
    for x in targets:
        ret_list.append(doc_list[x])

    return ret_list



if __name__ == "__main__":
    #files = glob.glob("./*.json")
    files = ["BioASQ-trainingDataset5b.json", "BioASQ-trainingDataset6b.json"]
    if len(files) == 0:
        print "no files to process"
        sys.exit(1)

    setup_tfidf()
    with open("neg_docs.txt", "w") as outfile:
        process_file(outfile, files)
