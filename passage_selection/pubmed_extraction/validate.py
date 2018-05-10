import sys

def parseBookXML(infiles):
    for infile in infiles:
        print "checking " + infile
        with open(infile, 'r') as f:
            for line in f:
                arr = line.split('\t')
                if len(arr) != 3:
                    print "error with elements on line: " + line

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "one file input arg required"
        sys.exit(1)

    parseBookXML(sys.argv[1:])

