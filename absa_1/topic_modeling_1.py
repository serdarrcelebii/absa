import pandas as pd
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
NUM_TOPICS = 10
WORDS_PER_TOPIC = 10
LDA_ITERATIONS = 1000 #20

# Build a Dictionary - association word to numeric id
dictionary = gensim.corpora.Dictionary(processed_docs)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in processed_docs]

# Build the LDA model
lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary,
                            passes=LDA_ITERATIONS)  # passes= 1 by default? -> MORE PASSES WILL INCREASE ACCURACY

# Build the LSI model
# lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

print("LDA Model:")

for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lda_model.print_topic(idx, WORDS_PER_TOPIC))

print("=" * 20)

### these printed topics are saved as  identified topics

## TESTING



