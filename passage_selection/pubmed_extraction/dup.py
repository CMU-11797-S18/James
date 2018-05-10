import glob
import cPickle as pickle

def parseBookXML():
    doc_tracker = {}

    with open("dicts_new", "rb") as f:
        doc_tracker = pickle.load(f)

    for src in glob.glob("output/out08/*.tsv"):
        outfile = "output/fixed_out08/" + src[13:]
        with open(src, 'r') as f:
            with open(outfile, 'w') as out:
                master_count = 0
                count = 0
                for line in f:
                    master_count += 1
                    arr = line.split('\t')
                    if arr[0] not in doc_tracker:
                        out.write(line)
                        doc_tracker[arr[0]] = True
                        count += 1
        print "Written " + str(count) + " records to " + outfile + ", deleted " + str(master_count - count) + " duplicate records"
    
    with open("dicts_new", "w") as f:
        f.write(pickle.dumps(doc_tracker))

if __name__ == "__main__":
    parseBookXML()

