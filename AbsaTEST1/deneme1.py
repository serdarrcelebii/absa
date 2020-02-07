import sys, string
import nltk
import numpy as np
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import re
import pandas as pd
stemmer = PorterStemmer()
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
# load all review texts
def load_file(file):
    reviews = []
    ratings = []
    data = pd.read_csv('test.csv', error_bad_lines=False);
    reviews=data['Review'][0:]
    ratings = data['Rating'][0:]
    return reviews, ratings


# print len(reviews), reviews[1]

def parse_to_sentence(reviews):
    review_processed = []
    actual = []
    global only_sent
    only_sent= []
    for r in reviews:
        sentences = nltk.sent_tokenize(r)
        actual.append(sentences)
        sent = []
        for s in sentences:
            # words to lower case
            s = s.lower()
            # remove punctuations and stopwords
            replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            s = s.translate(replace_punctuation)
            stop_words = list(stopwords.words('english'))
            additional_stopwords = ["'s", "...", "'ve", "``", "''", "'m", '--', "'ll", "'d"]
            # additional_stopwords = []
            stop_words = set(stop_words + additional_stopwords)
            # print stop_words
            # sys.exit()
            word_tokens = word_tokenize(s)
            s = [w for w in word_tokens if not w in stop_words]
            # Porter Stemmer
            stemmed = [stemmer.stem(w) for w in s]
            if len(stemmed) > 0:
                sent.append(stemmed)
        review_processed.append(sent)
        only_sent.extend(sent)
    return review_processed, actual, only_sent


# sent = parse_to_sentence(reviews)
# print len(sent), sent[2]

def create_vocab(sent):
    words = []
    for s in sent:
        words += s
    freq = FreqDist(words)
    vocab = []
    for k, v in freq.items():
        if v > 5:
            vocab.append(k)
    # Assign a number corresponding to each word. Makes counting easier.
    vocab_dict = dict(zip(vocab, range(len(vocab))))
    return vocab, vocab_dict


def get_aspect_terms(file, vocab_dict):
    global aspect_terms
    aspect_terms=[]
    w_notfound = []
    f = open(file, "r")
    for line in f:
        s = line.strip().split(",")
        stem = [stemmer.stem(w.strip().lower()) for w in s]
        # we store words by their corresponding number.
        # aspect = [vocab_dict[w] for w in stem]
        aspect = []
        for w in stem:
            if w in vocab_dict:
                aspect.append(w)
            else:
                w_notfound.append(w)
        aspect_terms.append(aspect)
    # We are only using one hotel review file, as we keep inceasing the number of files words not found will decrease.
    # print "Words not found in vocab:", ' '.join(w_notfound)
    f.close()
    return aspect_terms


# def chi_sq(w, A, sent):

def chi_sq(a, b, c, d):
    c1 = a
    c2 = b - a
    c3 = c - a
    c4 = d - b - c + a
    nc = d
    return nc * (c1 * c4 - c2 * c3) * (c1 * c4 - c2 * c3) / ((c1 + c3) * (c2 + c4) * (c1 + c2) * (c3 + c4))


def aspect_segmentaion():
    # Sentiment analysis
    sid = SIA()

    # INPUT
    # review, this algo needs all the review. Please process dataset.
    file = "Data/Texts/hotel_72572_parsed.txt"
    reviews, all_ratings = load_file(file)

    # selection threshold
    p = 5
    # Iterations
    # I = 10
    I = 1

    # Create Vocabulary
    review_sent, review_actual, only_sent = parse_to_sentence(reviews)
    global vocab
    vocab, vocab_dict = create_vocab(only_sent)

    # Aspect Keywords
    aspect_file = "aspect_keywords.csv"
    aspect_terms = get_aspect_terms(aspect_file, vocab_dict)

    label_text = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Check in/Front Desk', 'Service', 'Business Service']
    print (aspect_terms)

    # ALGORITHM
    review_labels = []
    k = len(aspect_terms)
    v = len(vocab)
    aspect_words = np.zeros((k, v))
    aspect_sent = np.zeros(k)
    num_words = np.zeros(v)

    for i in range(I):
        for r in review_sent:
            labels = []
            for s in r:
                count = np.zeros(len(aspect_terms))
                i = 0
                for a in aspect_terms:
                    for w in s:
                        if w in vocab_dict:
                            num_words[vocab_dict[w]] += 1
                            if w in a:
                                count[i] += 1
                    i = i + 1
                if max(count) > 0:
                    la = np.where(np.max(count) == count)[0].tolist()
                    labels.append(la)
                    for i in la:
                        aspect_sent[i] += 1
                        for w in s:
                            if w in vocab_dict:
                                aspect_words[i][vocab_dict[w]] += 1
                else:
                    labels.append([])
            review_labels.append(labels)

    print(review_labels)
    ratings_sentiment = []
    for r in review_actual:
        sentiment = []
        # aspect ratings based on sentiment
        for s in r:
            ss = sid.polarity_scores(s)
            sentiment.append(ss['compound'])
        ratings_sentiment.append(sentiment)
    print(ratings_sentiment)
    # Aspect Ratings Per Review
    aspect_ratings = []
    for i, r in enumerate(review_labels):
        rating = np.zeros(7)
        count = np.zeros(7)
        rs = ratings_sentiment[i]
        for j, l in enumerate(r):
            for k in range(7):
                if k in l:
                    rating[k] += rs[j]
            for k in range(7):
                if count[k] != 0:
                    rating[k] /= count[k]
        # Map from -[-1,1] to [1,5]
        print('rating', rating)
        for k in range(7):
            if rating[k] != 0:
                rating[k] = int(round((rating[k] + 1) * 5 / 2))
        aspect_ratings.append(rating)
    return aspect_ratings, all_ratings


print(aspect_segmentaion())


# n = 0
# print review_actual[n], '\n', review_labels[n]
# print ratings_sentiment[n], '\n', aspect_ratings[n]
# print len(all_ratings), len(reviews), all_ratings[0]
# sys.exit()
# return aspect_ratings

# print sent[5:9], labels[5:9]
# print zip(actual_sent, labels)[:10]
# print zip(actual_sent, sentiment)[:10]