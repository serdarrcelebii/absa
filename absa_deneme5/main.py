import pandas as pd
import numpy as np
import os
from itertools import compress
import nltk
import matplotlib.pyplot as plt
import string
import sys, string
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import re
import itertools
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
wnl = WordNetLemmatizer()
import copy

def parse_to_sentence(reviews):
    """
    Need to run parse_to_sentence func "the number of movies"'s time and join the parsing result for each movie together
    e.g. input: text_list[0]
    param idx: an integer between 0 - 38. movie.title[idx] shows the title of the movie
    e.g. movie.title[0] = 'Shazam!'
    param reviews: a list of reviews (string) FOR ONE MOVIE
    return review_processed: processed reviews from each user in a list named review_processed
    return actual: the sentence level tokenized reviews in a list named actual
    return only_sent: each processed sentence
    """
    # model of punctuation removal
    replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    # define stop words list for this movie
    stop_words = list(stopwords.words('english'))
    #stop_words=stop_words+
    # add very the movie title to the stop word (process the title before add it to the stop word list)

    # process the review text
    review_processed = []
    actual = []
    only_sent = []
    # for each review in the input review list
    print(reviews)
    for r in reviews['Review']:
        sentences = nltk.sent_tokenize(r)
        actual.append(sentences)
        sent = []
        # for each sentence in this review doc
        temp = []
        for s in sentences:
            # words to lower case
            s = s.lower()
            # remove punctuations
            s = s.translate(replace_punctuation)
            word_tokens = word_tokenize(s)
            # remove stopwords
            s = [w for w in word_tokens if not w in stop_words]
            s = [w for w in word_tokens if not w.isdigit()]
            # Porter Stemmer

            stemmed = [wnl.lemmatize(w) for w in s]
            if len(stemmed) > 0:
                sent.append(stemmed)
                temp.append(s)
        review_processed.append(temp)
        only_sent.extend(sent)
    return review_processed, actual, only_sent

data = pd.read_csv('test2.csv')
print(data.head())


A = []  # removed punctuation and stop words
B = []  # only tokenized to sentences
C = []  # removed punctuation, stop words, and stemmed
#for i in data.to_numpy():
a,b,c = parse_to_sentence(data)
A.append(a)
B.append(b)
C.append(c)
print(A)
print(B)
print(C)
print(len(A),len(B),len(C))

flat_C = list(itertools.chain.from_iterable(C))
X = copy.deepcopy(flat_C)  # X in STEP 0
print(len(flat_C))  # flat_C is a list contains all sentences from all reviews. there are len(flat_C) = 53217 sentences

flat_flat_C = list(itertools.chain.from_iterable(flat_C)) # futher flat the list to combine all processed word into one list
word, count = np.unique(flat_flat_C,return_counts = True)
vocabulary = pd.DataFrame([word, count]).T
vocabulary.columns  = ['word','count']
print(vocabulary.head())
print(vocabulary.tail())
vocal_f =  vocabulary.set_index('word')['count'].to_dict()
print(vocal_f)
vocal_w = dict(zip(range(len(vocabulary.word)),vocabulary.word))
print(vocal_w)
sorted_val = vocabulary.sort_values('count')
sorted_val.reset_index(drop = True, inplace = True)
print(sorted_val.head())
print(sorted_val[-100:])
with open('common_words.txt') as f:
    common_words = f.read().splitlines()
print(len(common_words)) # 1000
common_words[0] = 'the'
print(common_words[:5])
stemmed_common_wrods = [stemmer.stem(w) for w in common_words]
# len(stemmed_common_wrods)  # still 10000
print('cw: ',common_words[-10:])
print('\nstemmed cw: ',stemmed_common_wrods[-10:])
imprt_val = [x for x in list(sorted_val.word) if not x in stemmed_common_wrods[:550]]
print(len(imprt_val))
imprt_cnt = []
for v in imprt_val:
    imprt_cnt.append(vocal_f[v])
imprt_val_df = pd.DataFrame([imprt_val, imprt_cnt]).T
imprt_val_df.columns = ['word','count']
patch = pd.DataFrame([['long','minut','person','action'],[668,569,566,2351]]).T
patch.columns = ['word','count']
imprt_val = pd.concat([imprt_val_df, patch],ignore_index = True)
imprt_val = imprt_val.sort_values(by = 'count')
imprt_val.reset_index(drop = True,inplace = True)
print(imprt_val[-10:])
imprt_w = dict(zip(range(len(imprt_val.word)),imprt_val.word))
print(imprt_w)
imprt_f = imprt_val.set_index('word')['count'].to_dict()
print(imprt_f)

aspects = ['value','room','location','cleanliness','service']
seed1 = ['value', 'price' ]
seed2 = ['room','space'  ]
seed3 = [ "location", "locate" ]
seed4 = [ "clean", "dirty" ]
seed5 = [ 'service', 'manager' ]
seeds = [seed1, seed2, seed3, seed4, seed5]

# label each sentence with an aspect
def step1n2(keyword_list,sentence_list = flat_C):
    """
    Match aspect keywords in each sentence of X and record the matching hits for each aspect i

    param sentence_list: a list of sentence that will be compared with each aspect keywords
    param keyword_list: a list of lists where sublists contains keywords for different aspects

    return msa: "matched sentence & aspects". a list of lists (shape = len(sentence_list) * len(aspects)), where sublists are keyword match counts for sentences
    return l: a list of aspect label for each sentence (shape = len(sentence_list) * 1); 0 means no match; 1-5 indicates aspect label
    """
    msa = []
    l = []
    for sent in sentence_list:
        sa = []
        for keywords in keyword_list:
            sa.append(len(set(sent) & set(keywords)))
        msa.append(sa)

        if sum(sa) != 0:     # calculate the label if five aspect matches are not all zeros
            winner = np.argwhere(sa == np.amax(sa))+1  #valid winner value: 1,2,3,4,5
            l.append(winner.flatten().tolist())
        else:
            l.append([0])
    return msa, l

msa,labels = step1n2(seeds,sentence_list = flat_C)
print(labels[:5])
print(msa[0],labels[0])
#65 te kaldım

aspect_df = copy.deepcopy(imprt_val)


# https://blog.csdn.net/shenxiaoming77/article/details/51473986
def get_C(sentence_list=flat_C, aspects=aspects, labels=labels, aspect_df=aspect_df, imprt_w=imprt_w,
          imprt_val=imprt_val):
    """
    C1 is the number of times w occurs in sentences belonging to aspect Ai
    C2 is the number of times w occurs in sentences NOT belonging to Ai
    C3 is the number of sentences of aspect Ai that do not contain w
    C4 is the number of sentences that neither belong to aspect Ai , nor contain word w,

    return matrix_list: a list contains five matrices; each matrix contains all 4 C values for an aspect
    return a_index_list: a list of setence indices; sentence belong to a certain aspect
    return na_index_list: a list of setence indices; sentence NOT belong to a certain aspect
    """

    a_index_list = []  # list of lists; sublist0 contains sentence index that belongs to a certain aspect
    na_index_list = []  # list of lists; sublist0 contains sentence index that NOT belongs to a certain aspect
    matrix_list = []  # a list stores all 5 a_matrix, each of the matrix contains C 1,2,3,4 values for an aspect
    num_words = len(imprt_w)

    for a in range(len(aspects)):
        print('Calculating C values for aspect {}...'.format(a + 1))
        b_list = []
        for label in labels:
            b_list.append(a + 1 in label)  # b_list is a list contais True or False with a shape of len(all sentences)

        # get all sentences belong to aspect a
        a_index = list(compress(range(len(b_list)), b_list))
        na_index = [idx for idx in range(len(flat_C)) if not idx in a_index]
        sents_a = list(flat_C[i] for i in a_index)  # get all sentences belongs to aspect a
        sents_na = list(flat_C[i] for i in na_index)
        # ==============================================================================================================
        a_matrix = np.empty((num_words, 4))

        for w in range(num_words):  # check if w1 is (not) in s1, s2, ..., sn;
            word = imprt_w[w]
            # Calcuate C1, C3
            counter1 = 0  # counter1 cumsum value for C1
            for sa in sents_a:
                counter1 += int(word in sa)  # cumsum when word appears in sent_a
            c1 = counter1
            c3 = len(sents_a) - counter1

            # Calculate C2, C4
            counter2 = 0  # counter2 cumsum value for C2
            for nsa in sents_na:
                counter2 += int(word in nsa)  # cumsum when word appears in sent_na
            c2 = counter2
            c4 = len(sents_na) - counter2

            # fill numbers (C values) to a_matrix
            a_matrix[w, 0] = c1
            a_matrix[w, 1] = c2
            a_matrix[w, 2] = c3
            a_matrix[w, 3] = c4

        matrix_list.append(a_matrix)

        a_index_list.append(a_index)
        na_index_list.append(na_index)
    #         break

    return matrix_list, a_index_list, na_index_list

matrix_list, a_index_list, na_index_list = get_C(sentence_list = flat_C,aspects = aspects, labels = labels , aspect_df = aspect_df, imprt_val = imprt_val)
print(matrix_list)
a0_Cvalue_matrix =matrix_list[0]
a1_Cvalue_matrix =matrix_list[1]
a2_Cvalue_matrix =matrix_list[2]
a3_Cvalue_matrix =matrix_list[3]
a4_Cvalue_matrix =matrix_list[4]

a0_Cvalue_matrix = np.array(a0_Cvalue_matrix)
a1_Cvalue_matrix = np.array(a1_Cvalue_matrix)
a2_Cvalue_matrix = np.array(a2_Cvalue_matrix)
a3_Cvalue_matrix = np.array(a3_Cvalue_matrix)
a4_Cvalue_matrix = np.array(a4_Cvalue_matrix)
print(a1_Cvalue_matrix.shape,a2_Cvalue_matrix.shape,a3_Cvalue_matrix.shape,a4_Cvalue_matrix.shape)


matrix_list = [a0_Cvalue_matrix,a1_Cvalue_matrix,a2_Cvalue_matrix,a3_Cvalue_matrix,a4_Cvalue_matrix]
c_matrix  = np.array(matrix_list)
print(c_matrix.shape)

def chi(c, c1, c2, c3, c4):
    """
    The χ2 statistic to compute the dependencies between a term w and aspect Ai is defined as follows:

    param c : C  is the total number of word occurrences
    param c1: C1 is the number of times w occurs in sentences belonging to aspect Ai
    param c2: C2 is the number of times w occurs in sentences NOT belonging to Ai
    param c3: C3 is the number of sentences of aspect Ai that do not contain w
    param c4: C4 is the number of sentences that neither belong to aspect Ai , nor contain word w,

    return xwa: chi-squared value (float)
    """
    xwa = (c * (c1 * c4 - c2 * c3) ** 2) / ((c1 + c3) * (c2 + c4) * (c1 + c2) * (c3 + c4))
    return xwa


c_matrix = np.array(matrix_list)
# c_matrix.shape = (5, 16312, 4)
# create a new matrix to store all chi values; number of rows = number of words, number of columns = number of aspects
chi_matrix = np.empty((c_matrix.shape[1], c_matrix.shape[0]))
for w in range(c_matrix.shape[1]):
    word = imprt_w[w]
    print('\nCalculating Chi2 for word: {}'.format(word))
    c = imprt_f[word]
    for a in range(c_matrix.shape[0]):
        c1 = c_matrix[a, w, 0]
        c2 = c_matrix[a, w, 1]
        c3 = c_matrix[a, w, 2]
        c4 = c_matrix[a, w, 3]

        print('C values: ', c, c1, c2, c3, c4)
        xwa = chi(c, c1, c2, c3, c4)
        print('A{} Chi2: '.format(a), xwa)

        chi_matrix[w, a] = xwa


# chi_matrix has shape of w * a
def get_keywords(a, p, chi_matrix=chi_matrix, vocabulary=imprt_val.word, aspects=aspects):
    """
    param a: an integer that indicates which aspect you want to see (choose from range(aspects))
    param p: an integer that indicates how many words you want to obtain for this aspect
    param chi_matrix: a matrix with shape of w * a; each cell contains the chi2 dependency between the word and the aspect
    param vocabulary: default imprt_val.word indicates the unique vocabularies that we used to generate keywords. Word order should be the same as the word order in the matrix
    param aspects: default aspects, a list of strings indicates the asepcts (for printing purpose)

    return updated_keywords: an array contains p keywords for aspect a
    """
    chi_df = pd.DataFrame(chi_matrix[:, a])
    chi_df.columns = ['chi']
    sorted_chi = chi_df.sort_values(by='chi', ascending=False)
    top_word_idx = sorted_chi.index.tolist()[:p]
    top_word = vocabulary[top_word_idx]  # a dataframe with index and keywords
    print('Aspect: "{}":'.format(aspects[a]))
    print('  * Original Seed Keywords: {} '.format(seeds[a]))
    print('  * Sorted & Upated {} Keywords: {} \n'.format(p, top_word.values))
    return list(top_word.values)


# see the updated keyword lists:
k = []  # K is a list of five sublists; each sublist contains the updated keywords for that aspect
for i in range(len(aspects)):
    k.append(get_keywords(i, 10, chi_matrix = chi_matrix, vocabulary = imprt_val.word))

print(k)

flat_A = list(itertools.chain.from_iterable(A))  ###flat_A: one review; A: reviews for one movie###
# len(flat_A)   # 4085 reviews in  total

review_segs_m = []
review_segs_l = [] # segment each review to five different asepects. Word matches in m, overall label of aspect in l
for review in flat_A:
    m, l = step1n2(k,sentence_list = review)
    review_segs_m.append(m)
    review_segs_l.append(l)

print('There are {} sentences in the first review'.format(len(review_segs_l[2])))
print('There are {} labels for the third sentences in the first review'.format(len(review_segs_l[2][0])))
print('Those labels are {}'.format(review_segs_l[2][0]))

flat_A = list(itertools.chain.from_iterable(A))


def get_Wd(review, review_labels):
    """
    Get Wd for a review
    param review: a list of sentences representing a review
    param review_labels: a list of aspect label(s) for each sentence within this review
    return Wd: a k × n feature dataframe, where W dij is the frequency of word wj in the text assigned to aspect Ai of d normalized
    """
    # create dataframe of Wd
    word_list = list(
        itertools.chain.from_iterable(review))  # convert each review from a list of sentences to a list of words
    rwords = np.unique(word_list)
    Wd = pd.DataFrame(np.empty(len(rwords)))
    Wd.index = rwords

    ## group sentences in review to five aspects
    for a in range(len(aspects) + 1):
        #         print(a)
        if a == 0:  # when this sentence doesn't belong to any of the five aspects
            df = pd.DataFrame([np.nan] * len(rwords))
            df.index = rwords
        else:
            b_index = [a in sublist for sublist in review_labels]
            sent_index = list(compress(range(len(b_index)), b_index))  # sentence index that belong to aspect a
            #             print('b index, sent index: ',b_index, sent_index)
            a_sentences = [review[i] for i in sent_index]
            a_words = list(itertools.chain.from_iterable(a_sentences))
            ## compute word frequencies within this aspect, and then normalize the frequency by the toal words in this aspect
            unique_word, word_freq = np.unique(a_words, return_counts=True)
            norm_freq = word_freq / len(a_words)
            # create a dataframe of unique words and norm freq, merge this dataframe to Wd dataframe by columns
            df = pd.DataFrame(norm_freq)
            df.index = unique_word
        Wd = pd.concat([Wd, df], axis=1)
    # when the Wd matrix is finished,define the column names (aspects), drop the first columns (empty col), then transpose the dataframe to match the shape in Prof. Paper
    Wd.columns = [0, 'NAN', 'a1', 'a2', 'a3', 'a4', 'a5']
    Wd = Wd.drop(0, axis=1)
    Wd = Wd.T
    return Wd

Wd_list = []
for r in range(len(flat_A)):
    Wd_list.append(get_Wd(flat_A[r], review_segs_l[r]))

print(len(Wd_list))
print(np.nansum(Wd_list[3].loc['a4']))
print(np.array(Wd_list[3].iloc[2]))




def get_aspect_rating(Wd):
    """
    Calculate rating for each aspect for a processed review
    param Wd: processed review
    return ratings: an array with rating for each aspect (total five)
    """
    polarities = []
    for word in Wd.columns:
        polarities.append(sia.polarity_scores(word)['compound'])
    polarities = np.array(polarities)

    ratings = np.empty((5))
    for a in range(1, 6):
        word_freq = np.array(Wd.iloc[a])
        rating = np.nansum(word_freq * polarities)
        ratings[a - 1] = rating

    return ratings

aspect_ratings = np.empty((len(Wd_list),5))
for i in range(len(Wd_list)):
    Wd = Wd_list[i]
    aspect_ratings[i] = get_aspect_rating(Wd=Wd)

print(aspect_ratings.shape)

print('Aspect 1-5 Ratings')
print(aspects)
aspect_rating_df = pd.DataFrame(aspect_ratings)
print(aspect_rating_df.head())


