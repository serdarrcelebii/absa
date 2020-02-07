from py2neo import Graph
import re, string
from normalization import *

uri = "bolt://localhost:7687"
user = "neo4j"
password = "Odeon.12"

graphdb = Graph(uri=uri, user=user, password=password)
# define some parameterized Cypher queries

# For data insertion
INSERT_QUERY = '''
    FOREACH (t IN {wordPairs} | 
        MERGE (w0:Word {word: t[0]})
        MERGE (w1:Word {word: t[1]})
        CREATE (w0)-[:NEXT_WORD]->(w1)
        )
'''

# get the set of words that appear to the left of a specified word in the text corpus
LEFT1_QUERY = '''
    MATCH (s:Word {word: {word}})
    MATCH (w:Word)-[:NEXT_WORD]->(s)
    RETURN w.word as word
'''

# get the set of words that appear to the right of a specified word in the text corpus
RIGHT1_QUERY = '''
    MATCH (s:Word {word: {word}})
    MATCH (w:Word)<-[:NEXT_WORD]-(s)
    RETURN w.word as word
'''

regex = re.compile('[%s]' % re.escape(string.punctuation))
exclude = set(string.punctuation)

def arrifySentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.strip()
    sentence = regex.sub('', sentence)
    wordArray = sentence.split()
    tupleList = []
    for i, word in enumerate(wordArray):
        if i+1 == len(wordArray):
            break
        tupleList.append([word, wordArray[i+1]])
    return tupleList


def loadFile():

    tx = graphdb.begin()
    data=get_data('test.csv')
    corpus=normalized_corpus(data['Review'])
    count = 0
    for l in corpus:
        print(l)
        params = {'wordPairs': arrifySentence(l)}
        tx.run(INSERT_QUERY, params)
        tx.process()
        count += 1
        if count > 300:
            tx.commit()
            tx = graphdb.begin()
            count = 0
    # with open('data/ceeaus.dat', encoding='ISO-8859-1') as f:
    #     count = 0
    #     for l in f:
    #         print(l)
    #         params = {'wordPairs': arrifySentence(l)}
    #         tx.run(INSERT_QUERY, params)
    #         tx.process()
    #         count += 1
    #         if count > 300:
    #             tx.commit()
    #             tx = graphdb.begin()
    #             count = 0
    # f.close()
    tx.commit()
#loadFile()
LEFT1_QUERY = '''
    MATCH (s:Word {word: {word}})
    MATCH (w:Word)-[:NEXT_WORD]->(s)
    RETURN w.word as word
'''

# get the set of words that appear to the right of a specified word in the text corpus
RIGHT1_QUERY = '''
    MATCH (s:Word {word: {word}})
    MATCH (w:Word)<-[:NEXT_WORD]-(s)
    RETURN w.word as word
'''

# return a set of all words that appear to the left of `word`
def left1(word,tx):
    params = {
        'word': word.lower()
    }

    results = tx.run(LEFT1_QUERY, params)

    words = []
    for result in results:
        for line in result:
            words.append(line)

    return set(words)

# return a set of all words that appear to the right of `word`
def right1(word,tx):
    params = {
        'word': word.lower()
    }

    results = tx.run(RIGHT1_QUERY, params)
    words = []
    for result in results:
        for line in result:
            words.append(line)

    return set(words)

# compute Jaccard coefficient
def jaccard(a,b):
    intSize = len(a.intersection(b))
    unionSize = len(a.union(b))
    return intSize / unionSize

# we define paradigmatic similarity as the average of the Jaccard coefficents of the `left1` and `right1` sets
def paradigSimilarity(w1, w2):
    graphdb = Graph(uri=uri, user=user, password=password)
    tx = graphdb.begin()
    asd=(jaccard(left1(w1,tx), left1(w2,tx)) + jaccard(right1(w1,tx), right1(w2,tx))) / 2.0
    tx.commit()
    return asd

