import pandas as pd
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
import gensim
from gensim import corpora
import math

class BM25 :
    def __init__(self, fn_docs, delimiter='|') :
        self.dictionary = corpora.Dictionary()
        self.DF = {}
        self.delimiter = delimiter
        self.DocTF = []
        self.DocIDF = {}
        self.N = 0
        self.DocAvgLen = 0
        self.fn_docs = fn_docs
        self.DocLen = []
        self.buildDictionary()
        print(self.dictionary.items())
        self.TFIDF_Generator()

    def buildDictionary(self) :
        raw_data = []
        for line in self.fn_docs :
            raw_data.append(line.strip().split(self.delimiter))
        print(raw_data)
        self.dictionary.add_documents(raw_data)

    def TFIDF_Generator(self, base=math.e) :
        docTotalLen = 0
        for line in (self.fn_docs) :
            doc = line.strip().split(self.delimiter)
            docTotalLen += len(doc)
            self.DocLen.append(len(doc))
            #print self.dictionary.doc2bow(doc)
            bow = dict([(term, freq*1.0/len(doc)) for term, freq in self.dictionary.doc2bow(doc)])
            for term, tf in bow.items() :
                if term not in self.DF :
                    self.DF[term] = 0
                self.DF[term] += 1
            self.DocTF.append(bow)
            self.N = self.N + 1
        for term in self.DF:
            self.DocIDF[term] = math.log((self.N - self.DF[term] +0.5) / (self.DF[term] + 0.5), base)
        self.DocAvgLen = docTotalLen / self.N

    def BM25Score(self, Query=[], k1=1.5, b=0.75) :
        query_bow = self.dictionary.doc2bow(Query)
        scores = []
        for idx, doc in enumerate(self.DocTF) :
            commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
            tmp_score = []
            doc_terms_len = self.DocLen[idx]
            for term in commonTerms :
                upper = (doc[term] * (k1+1))
                below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.DocAvgLen))
                tmp_score.append(self.DocIDF[term] * upper / below)
            scores.append(sum(tmp_score))
        return scores

    def TFIDF(self) :
        tfidf = []
        for doc in self.DocTF :
            doc_tfidf  = [(term, tf*self.DocIDF[term]) for term, tf in doc.items()]
            doc_tfidf.sort()
            tfidf.append(doc_tfidf)
        return tfidf

    def Items(self) :
        # Return a list [(term_idx, term_desc),]
        items = self.dictionary.items()
        return items



lemmatizer = WordNetLemmatizer()
def lemma_tokens(tokens, stemmer):
    lemma = []
    for item in tokens:
        lemma.append(lemmatizer.lemmatize(item))
    return lemma

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    stems = lemma_tokens(tokens, stemmer)
    stems = [w for w in stems if len(w) > 2]
    return stems

data_file_name = 'test.csv'


data_df = pd.read_csv(data_file_name)
vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 85
)

corpus_data_features = vectorizer.fit_transform(data_df['Review'].tolist())
bm25 = BM25(data_df['Review'], delimiter=' ')
print(bm25)
Query = 'value price money quality deal hotel package rate resort budget ticket accommodation amount credit card dollar building travel discount agent luxury corner'
Query = Query.split()
print(Query)
scores = bm25.BM25Score(Query)
print(scores)
for i,item in enumerate(scores):
    print(i,item)
tfidf = bm25.TFIDF()
for i, tfidfscore in enumerate(tfidf):
    print (i, tfidfscore)

