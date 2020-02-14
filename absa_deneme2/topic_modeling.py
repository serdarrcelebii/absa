from gensim import corpora, models
import gensim
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from utils import *
from preprocess import *

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import tqdm
def print_topics_gensim(topic_model, total_topics=1,weight_threshold=0.0001, display_weights=False,num_terms=None):
    for index in range(total_topics):
        topic = topic_model.show_topic(index)
        topic = [(word, round(wt,2))
        for word, wt in topic if abs(wt) >= weight_threshold]
        if display_weights:
            print('Topic #'+str(index+1)+' with weights')
            print (topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print( tw[:num_terms] if num_terms else tw)

# get topics with their terms and weights
def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights,sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])
    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]
    return topics
# print all the topics from a corpus

def print_topics_udf(topics, total_topics=1, weight_threshold=0.0001,display_weights=False,num_terms=None):
    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt)) for term, wt in topic]
        topic = [(word, round(wt,2)) for word, wt in topic  if abs(wt) >= weight_threshold]
    if display_weights:
        print ('Topic #'+str(index+1)+' with weights')
        print (topic[:num_terms] if num_terms else topic)
    else:
        print ('Topic #'+str(index+1)+' without weights')
        tw = [term for term, wt in topic]
        print (tw[:num_terms] if num_terms else tw)
    print

def train_lsi_model_gensim(corpus, total_topics=2):
    norm_tokenized_corpus = corpus
    dictionary = corpora.Dictionary(norm_tokenized_corpus)
    mapped_corpus = [dictionary.doc2bow(text) for text in norm_tokenized_corpus]
    tfidf = models.TfidfModel(mapped_corpus)
    corpus_tfidf = tfidf[mapped_corpus]
    lsi = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=total_topics)
    return lsi

def train_lda_model_gensim(corpus, total_topics=2):
    norm_tokenized_corpus = corpus
    dictionary = corpora.Dictionary(norm_tokenized_corpus)
    mapped_corpus = [dictionary.doc2bow(text) for text in norm_tokenized_corpus]
    tfidf = models.TfidfModel(mapped_corpus)
    corpus_tfidf = tfidf[mapped_corpus]
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary,iterations=1000,num_topics=total_topics)
    #print_coherence(lda,corpus,dictionary)
    return lda

def train_lda_sklearn(norm_corpus,total_topics):
    vectorizer, tfidf_matrix = build_feature_matrix(norm_corpus,feature_type = 'tfidf')
    feature_names = vectorizer.get_feature_names()
    print(feature_names)
    lda = LatentDirichletAllocation(n_components=total_topics,max_iter = 100, learning_method = 'online', learning_offset = 50.,random_state = 42)
    lda.fit(tfidf_matrix)
    weights = lda.components_
    topics = get_topics_terms_weights(weights, feature_names)
    print_topics_udf(topics=topics,total_topics = total_topics)


def coherence_score(lda,corpus,dictionary):
    coherence_model_lda = CoherenceModel(model=lda, texts=corpus, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

def compute_coherence_values(corpus, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        norm_tokenized_corpus = corpus
        dictionary = corpora.Dictionary(norm_tokenized_corpus)
        mapped_corpus = [dictionary.doc2bow(text) for text in norm_tokenized_corpus]
        tfidf = models.TfidfModel(mapped_corpus)
        corpus_tfidf = tfidf[mapped_corpus]
        model= models.LdaModel(corpus_tfidf, id2word=dictionary,num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=corpus, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def print_coherence(corpus):
    # Show graph
    model_list, coherence_values = compute_coherence_values(corpus=corpus, start=2,limit=20, step=6)
    limit = 20;
    start = 2;
    step = 6;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print( "Topic %d:" % (topic_idx))
        print( " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def train_nmf_model(corpus,no_topics = 7,no_features = 1000):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(corpus)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

    no_top_words = 7
    display_topics(nmf, tfidf_feature_names, no_top_words)

