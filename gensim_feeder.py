from gensim import corpora, models, similarities
from six import iteritems
import csv
import string

def tfidf(filename):
    with open(filename, newline='') as csvfile:
        csv.field_size_limit(1000000000)
        csv_filename = csv.reader(csvfile, quotechar='"')
        dictionary = corpora.Dictionary(token(''.join(row)) for row in csv_filename)
    stoplist = set('a about above after again against all am and any are \
                arent as at be because been before being below between \
                both but by cant cannot could couldnt did didnt do does \
                doing dont down during each few for from further had hadnt \
                has hasnt have havent having he hed hes her here heres \
                hers herself him himself his how hows i id im ive if in \
                into is isnt it its its itself me more most mustnt my \
                myself no nor not of off on once only or other ought our \
                ours ourselves out over own same shant she shes should \
                shouldnt so some such than that thats the their theirs \
                them themselves then there theres these they theyd theyll \
                theyre theyve this through to too until up very wasnt we \
                weve were werent what whats when whens where wheres which \
                while who whos whom why whys with wont would wouldnt you \
                youd youll youre youve your yours yourself yourselves'.split())
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                    if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed

    with open(filename, newline='') as csvfile:
        csv.field_size_limit(1000000000)
        csv_filename = csv.reader(csvfile, quotechar='"')
        corpus = [dictionary.doc2bow(token(''.join(row))) for row in csv_filename]

    tfidf = models.TfidfModel(corpus)

    lsi = models.LsiModel(tfidf[corpus], id2word=dictionary, num_topics=100)
    for topic in lsi.show_topics(num_topics=10):
        print(topic)
        print()

    lda = models.LdaModel(tfidf[corpus], id2word=dictionary, num_topics=100)
    for topic in lda.show_topics(num_topics=10):
        print(topic)
        print()

def main():
    tfidf('state-of-the-union.csv')
    print()
    tfidf('ap.csv')

def token(str):
    return str.translate(str.maketrans('','', string.punctuation)).lower().split()

if __name__ == "__main__":
    main()
