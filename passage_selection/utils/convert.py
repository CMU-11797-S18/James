## marges types.txt and vectors.txt into one file that follows
## the gensim format for reading with KeyedVectors.load_word2vec_format()

# number of lines (and vocab size) in the files
NUM_LINES = 1701632
EMBED_DIMS = 200

with open("types.txt", "r") as types, open("vectors.txt", "r") as vectors, open("combined.txt", "w") as combined:
    combined.write(str(NUM_LINES) + " " + str(EMBED_DIMS) + "\n")
    for x in xrange(NUM_LINES):
        word = types.readline().strip()
        vect = vectors.readline().strip()
        combined.write(word + " " + vect + "\n")
