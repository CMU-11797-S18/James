import json
from utils import get_qa_pair

with open('./BioASQ-trainingDataset5b.txt', encoding='utf8') as f:
    data = json.load(f)

print('getting qa pair from data')
qa_pair = get_qa_pair(data)
print(qa_pair[0])

print('adding spans into the qa_pair')
qa_pair = add_span(qa_pair)
print(qa_pair[0])
print(qa_pair[0][1])
for (snippet, spans) in zip(qa_pair[0][3], qa_pair[0][4]):
    print(spans)
    print(snippet)
    for (start, end) in spans:
        print(snippet[start:end + 1])

print('getting bio word dictionary')
bio_wdict = get_word_dict(qa_pair, bioclean)

print('loading embedding vector for words')
(word_vec_bio, word2ind_bio) = load_embed_bioasq(bio_wdict, vec_path='./vectors.txt', type_path='./types.txt')
ind2word_bio = dict((v, k) for k, v in word2ind_bio.items())
bio_qa_ind = word_to_ind(word2ind_bio, qa_pair, bioclean)

from sklearn.model_selection import train_test_split

train, test = train_test_split(bio_qa_ind, test_size=0.2, random_state=32)
train, validate = train_test_split(train, test_size=0.2, random_state=32)
print('flatten span list')
train_flat = flattenSpanList(train)
test_flat = flattenSpanList(test)
validate_flat = flattenSpanList(validate)
bio_qa_ind_flat = flattenSpanList(bio_qa_ind)
print(bio_qa_ind_flat[0])
