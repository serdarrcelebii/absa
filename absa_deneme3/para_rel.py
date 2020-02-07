import math

import pandas as pd
import freeze
data = pd.read_csv('test.csv', error_bad_lines=False);
data_text = data[['Review']]
data_text['index'] = data_text.index
documents = data_text
print(data_text)
print(documents)
print(len(documents))
print(documents[:5])

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
nltk.download('wordnet')
print(WordNetLemmatizer().lemmatize('went', pos='v'))
stemmer = SnowballStemmer('english')
original_words = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned',
           'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational',
           'traditional', 'reference', 'colonizer','plotted']
singles = [stemmer.stem(plural) for plural in original_words]
print(pd.DataFrame(data = {'original word': original_words, 'stemmed': singles}))

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

doc_sample = documents[documents['index'] == 88].values[0][0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = documents['Review'].map(preprocess)

print(processed_docs[:10])

from collections import defaultdict

inverted_index = defaultdict(set)

for docid, terms in processed_docs.items():
     for term in terms:
          inverted_index[term].add(docid)

print(inverted_index['nice'])

# Building a TF-IDF representation using BM25

NO_DOCS = len(processed_docs)  # Number of documents

AVG_LEN_DOC = sum([len(doc) for doc in processed_docs]) / len(processed_docs)  # Average length of documents


# The function below takes the documentid, and the term, to calculate scores for the tf and idf
# components, and multiplies them together.
def tf_idf_score(k1, b, term, docid):
     ft = len(inverted_index[term])
     term = stemmer.stem(term.lower())
     fdt = processed_docs[docid].count(term)

     idf_comp = math.log((NO_DOCS - ft + 0.5) / (ft + 0.5))

     tf_comp = ((k1 + 1) * fdt) / (k1 * ((1 - b) + b * (len(processed_docs[docid]) / AVG_LEN_DOC)) + fdt)

     return idf_comp * tf_comp


# Function to create tf_idf matrix without the query component
def create_tf_idf(k1, b):
     tf_idf = defaultdict(dict)
     for term in set(inverted_index.keys()):
          for docid in inverted_index[term]:
               tf_idf[term][docid] = tf_idf_score(k1, b, term, docid)
     return tf_idf

tf_idf = create_tf_idf(1.5,0.5)


# Function to retrieve query component
def get_qtf_comp(k3, term, fqt):
     return ((k3 + 1) * fqt[term]) / (k3 + fqt[term])


# Function to retrieve documents || Returns a set of documents and their relevance scores.
def retr_docs(query, result_count):
     q_terms = [stemmer.stem(term.lower()) for term in query.split() if
                term not in gensim.parsing.preprocessing.STOPWORDS]  # Removing stopwords from queries
     fqt = {}
     for term in q_terms:
          fqt[term] = fqt.get(term, 0) + 1

     scores = {}

     for word in fqt.keys():
          # print word + ': '+ str(inverted_index[word])
          for document in inverted_index[word]:
               scores[document] = scores.get(document, 0) + (
                            tf_idf[word][document] * get_qtf_comp(0, word, fqt))  # k3 chosen as 0 (default)

     return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:result_count]

print(retr_docs("service",5))
print(processed_docs[20])