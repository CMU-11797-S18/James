## writes questions, exact answers, and snippets to a file. one line per item.
## reads json files in current directory as inputs

import json
from collections import defaultdict
import glob
import sys

questions = defaultdict(int)
#line_count = 0
def process_file(outfile, file_list):
    for infile in file_list:
        with open(infile, "r") as f:
            d = json.load(f)
            for item in d['questions']:
                outfile.write(item['body'].strip().replace('\n', ' ').encode("utf-8") + '\n')
                if 'exact_answer' in item:
                    for ans in item['exact_answer']:
                        if type(ans) is list:
                            for a in ans:
                                outfile.write(a.strip().replace('\n', ' ').encode("utf-8") + '\n')
                        else:
                            outfile.write(ans.strip().replace('\n', ' ').encode("utf-8") + '\n')
                if 'snippets' in item:
                    for s in item['snippets']:
                        sn = s['text'].strip().replace('\n', ' ').encode("utf-8")
                        outfile.write(sn + '\n')


if __name__ == "__main__":
    files = glob.glob("./*.json")
    #files = ["BioASQ-trainingDataset5b.json"]
    if len(files) == 0:
        print "no files to process"
        sys.exit(1)

    with open("all_docs.txt", "w") as outfile:
        process_file(outfile, files)
