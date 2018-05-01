## gather stats on lengths of questions and snippets from BioASQ training data
## and dump histogram data to file

import json
from collections import defaultdict

snip_data = defaultdict(int)
with open("BioASQ-trainingDataset5b.json", "r") as f:
    d = json.load(f)
    pairs = 0
    for item in d['questions']:
        pairs += len(item['snippets'])
        for s in item['snippets']:
            arr = s['text'].split()
            snip_data[len(arr)] += 1

with open("hist.json", "w") as f:
    json.dump(snip_data, f)
