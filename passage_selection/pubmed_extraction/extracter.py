from lxml import etree
from StringIO import StringIO
import glob
import cPickle as pickle
import os.path

def parseBookXML():
    doc_tracker = {}
    master_count = 0

    if os.path.exists("dicts"):
        with open("dicts", "rb") as f:
            doc_tracker = pickle.load(f)

    for src in glob.glob("/home/ec2-user/data/*.xml"):
        outfile = "/home/ec2-user/output/" + src[20:-4]+ "_out.tsv"
        with open(src, 'r') as f:
            xml = f.read()

        count = 0
        with open(outfile, 'w') as f:
            tree = etree.parse(StringIO(xml))
            context = etree.iterparse(StringIO(xml))
            title_str = ''
            abstract_str = ''
            pmid_str = ''
            del_count = 0
            for action, elem in context:
                if not elem.text:
                    text = "None"
                else:
                    text = elem.text
                if elem.tag == "ArticleTitle":
                    title_str = text
                if elem.tag == "AbstractText":
                    abstract_str = text
                if elem.tag == "PMID":
                    pmid_str = text
                if elem.tag == "PubmedArticle":
                    if pmid_str not in doc_tracker:
                        doc_tracker[pmid_str] = True
                        if abstract_str != '' and abstract_str != "None":
                            if count > 0:
                                f.write('\n')
                            f.write(pmid_str.encode("utf-8") + '\t' + title_str.encode("utf-8") + '\t' 
                                    + abstract_str.encode("utf-8"))
                            count += 1

                    else:
                        del_count += 1
                    title_str = ''
                    abstract_str = ''
                    pmid_str = ''

        master_count += count
        print "Written " + str(count) + " records to " + outfile + ", deleted " + str(del_count) + " duplicate records"

        '''k.key = outfile
        try:
            k.set_contents_from_filename(outfile)
            print "Written " + outfile + " to S3"
        except Exception, e:
            print "Error writing " + outfile + " to S3" '''

    
    with open("dicts", "w") as f:
        f.write(pickle.dumps(doc_tracker))

    print "Total records written: " + str(master_count)

if __name__ == "__main__":
    parseBookXML()

