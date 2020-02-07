import heapq
import json
from collections import defaultdict
from functools import partial

from nltk import FreqDist
import json
import nltk
from nltk.corpus import stopwords
import os
import glob
from nltk.stem.porter import *
path="hotelReviews/"
projectSettings="settings/"
from collections import defaultdict
import string
import numpy as np
aspect_list=['room', 'hotel', 'good', 'great', 'bed', 'clean', 'comfortable', 'desk', 'quiet', 'lot', 'spacious', 'elevator', 'evening', 'excellent', 'club', 'service', 'stay', 'location', 'staff', 'restaurant', 'view', 'large', 'choice', 'value', '\n', 'breakfast', 'thing', 'downtown', 'recommendation', 'bathroom', 'suite', 'size', 'standard', 'year', 'able', 'night', 'day', 'valet', 'arrange', 'car', 'person', 'second', '100', 'minute', 'line', 'match', 'bus', 'perfect', 'morning', 'weekend', 'people', 'spent', 'checked', 'avoid', 'march', 'spend', 'ave', 'lowestpriced', 'wish', 'romantic', 'child', 'time', 'inn', 'book', 'pike', 'husband', 'friend', 'feel', 'unbeatable', 'ace', 'experience', 'deal', 'review', 'convenient', 'recommend', 'design', 'centre', 'nice', 'friendly', 'bad', 'heard', 'noisy', 'pay', 'couch', 'helpful', 'arrival', 'property', 'drop', 'speak', 'street', 'family', 'food', 'luggage', 'market', 'today', 'double', 'westin', 'fine', 'unbelievable', 'downstairs', 'wrong', 'safe', 'towel', 'king', 'possible', 'expectation', 'basement', 'level', 'stairwell', 'come', 'dirty', 'adjacent', 'nyc', 'enter', 'style', 'lousy', 'luxury', 'floor', 'rate', 'available', 'nonsmoking', 'corner', 'block', 'business', 'right', 'new', 'hope', 'bath', 'whirlpool', 'feature', 'plenty', 'single', 'smart', 'lovely', 'robe', 'marble', 'watch']


class Review:
    def __init__(self):
        self.sentences = []  # list of objects of class Sentence
        self.reviewId = ""
        self.ratings = """{}"""  # true ratings provided by the user

    def __str__(self):
        retStr = ""
        for sentence in self.sentences:
            retStr += sentence.__str__() + '\n'
        retStr += "###" + self.reviewId + "###" + str(self.ratings) + "\n"
        return retStr

class Sentence:
    def __init__(self, wordList):
        self.wordFreqDict = FreqDist(wordList)#Dictionary of words in the sentence and corres. frequency
        self.assignedAspect = [] #list of aspects assigned to this sentence
    def __str__(self):
        return self.wordFreqDict.pformat(10000) + '##' + str(self.assignedAspect)

class ReadData:
    def __init__(self):
        self.aspectKeywords = []  # aspect name <--> keywords list
        self.stopWords = []
        self.wordFreq = {}  # dict with of all words and their freq in the corpus
        self.lessFrequentWords = set()  # words which have frequency<5 in the corpus
        self.allReviews = []  # list of Review objects from the whole corpus
        self.aspectSentences = defaultdict(list)  # aspect to Sentences mapping
        self.wordIndexMapping = {}  # word to its index in the corpus mapping
        self.aspectIndexMapping = {}  # aspect to its index in the corpus mapping

    def createWordIndexMapping(self):
        i = 0
        for word in self.wordFreq.keys():
            self.wordIndexMapping[word] = i
            i += 1
        print(self.wordIndexMapping)

    def readAspectSeedWords(self):
        aspectkeywordss=[]
        for aspect in aspect_list:
            print(aspect)
            list=[]
            list.append(aspect)
            self.aspectKeywords.append((aspect, list))
        print(self.aspectKeywords)

    def createAspectIndexMapping(self):
        i = 0;
        for aspect in self.aspectKeywords.keys():
            self.aspectIndexMapping[aspect] = i
            i += 1
        print(self.aspectIndexMapping)

    def stemmingStopWRemoval(self, review, vocab):
        ''' Does Following things:
        1. Tokenize review into sentences, and then into words
        2. Remove stopwords, punctuation and stem each word
        3. Add words into vocab
        4. Make Sentence objects and corresponding Review object
        '''
        reviewObj = Review()
        # copying ratings into reviewObj

        reviewObj.ratings = (review[1]['Rating'])
        reviewObj.reviewId=review[1]['S.No.']

        stemmer = PorterStemmer()
        reviewContent = review[1]['Review']
        # print(reviewContent)
        # TODO: Append title too!

        wordList=[]
        for word in reviewContent:
            vocab.append(word)
            wordList.append(word)
            if wordList:
                sentenceObj = Sentence(wordList)
                reviewObj.sentences.append(sentenceObj)
        if reviewObj.sentences:
            self.allReviews.append(reviewObj)
            # print(reviewObj)

    def readReviewsFromJson(self,data):
        ''' Reads reviews frm the corpus, calls stemmingStopWRemoval
        and creates list of lessFrequentWords (frequency<5)
        '''
        vocab = []
        count=0
        for reviews in data.iterrows():
            if count!=0:
                self.stemmingStopWRemoval(reviews, vocab)
            count=count+1
        self.wordFreq = FreqDist(vocab)
        for word, freq in self.wordFreq.items():
            if freq < 5:
                self.lessFrequentWords.add(word)
        for word in self.lessFrequentWords:
            del self.wordFreq[word]
        self.createWordIndexMapping()

        # print("Less Frequent Words ",self.lessFrequentWords)
        # print("Vocab ", self.wordFreq.pformat(10000))

    def removeLessFreqWords(self):
        emptyReviews = set()
        for review in self.allReviews:
            emptySentences = set()
            for sentence in review.sentences:
                deleteWords = set()
                for word in sentence.wordFreqDict.keys():
                    if word in self.lessFrequentWords:
                        deleteWords.add(word)
                for word in deleteWords:
                    del sentence.wordFreqDict[word]
                if not sentence.wordFreqDict:
                    emptySentences.add(sentence)
            review.sentences[:] = [x for x in review.sentences if x not in emptySentences]
            if not review.sentences:
                emptyReviews.add(review)
        self.allReviews[:] = [x for x in self.allReviews if x not in emptyReviews]

class BootStrap:
    def __init__(self, readDataObj):
        self.corpus = readDataObj
        # Aspect,Word -> freq matrix - frequency of word in that aspect
        self.aspectWordMat = defaultdict(lambda: defaultdict(int))
        # Aspect --> total count of words tagged in that aspect
        # = sum of all row elements in a row in aspectWordMat matrix
        self.aspectCount = defaultdict(int)
        # Word --> frequency of jth tagged word(in all aspects)
        # = sum of all elems in a column in aspectWordMat matrix
        self.wordCount = defaultdict(int)

        # Top p words from the corpus related to each aspect to update aspect keyword list
        self.p = 5
        self.iter = 7

        # List of W matrix
        self.wList = []
        # List of ratings Dictionary belonging to review class
        self.ratingsList = []
        # List of Review IDs
        self.reviewIdList = []

        '''def calcC1_C2_C3_C4(self):
            for aspect, sentence in self.corpus.aspectSentences.items():
                for sentence in sentences:
                    for word in self.corpus.wordFreq.keys() and not in sentence.wordFreqDict.keys():
                        self.aspectNotWordMat[aspect][word]+=1
                    for word,freq in sentence.wordFreqDict.items():
                        self.aspectWordMat[aspect][word]+=freq
        '''

    def assignAspect(self, sentence):  # assigns aspects to sentence
        sentence.assignedAspect = []
        count = defaultdict(int)  # count used for aspect assignment as in paper
        # print("IN ASSIGN ASPECT FUNCTION:",len(sentence.wordFreqDict))
        # print(sentence.wordFreqDict.keys())
        # print(self.corpus.aspectKeywords)
        for word in sentence.wordFreqDict.keys():
            for aspect, keywords in self.corpus.aspectKeywords:
                if word in keywords:
                    count[aspect] += 1
        if count:  # if count is not empty
            maxi = max(count.values())
            for aspect, cnt in count.items():
                if cnt == maxi:
                    sentence.assignedAspect.append(aspect)
        if (len(sentence.assignedAspect) == 1):  # if only 1 aspect assigned to it
            self.corpus.aspectSentences[sentence.assignedAspect[0]].append(sentence)

    def populateAspectWordMat(self):
        self.aspectWordMat.clear()
        for aspect, sentences in self.corpus.aspectSentences.items():
            for sentence in sentences:
                for word, freq in sentence.wordFreqDict.items():
                    print(aspect,word,freq)
                    self.aspectWordMat[aspect][word] += freq
                    self.aspectCount[aspect] += freq
                    self.wordCount[word] += freq

    def chiSq(self, aspect, word):
        # Total number of (tagged) word occurrences
        C = sum(self.aspectCount.values())
        C1 = 0
        # Frequency of word W in sentences tagged with aspect Ai
        try:
            C1 = self.aspectWordMat[aspect][word]
        except:
            C1=0
        # Frequency of word W in sentences NOT tagged with aspect Ai
        C2 = self.wordCount[word] - C1
        C3=0
        # Number of sentences of aspect A, NOT contain W
        try:
            C3 = self.aspectCount[aspect] - C1
        except:
            C3=0
        # Number of sentences of NOT aspect A, NOT contain W
        C4 = C - C1

        deno = (C1 + C3) * (C2 + C4) * (C1 + C2) * (C3 + C4)
        # print(aspect, word, C, C1, C2, C3, C4)
        if deno != 0:
            return (C * (C1 * C4 - C2 * C3) * (C1 * C4 - C2 * C3)) / deno
        else:
            return 0.0

    def calcChiSq(self):
        topPwords = []
        print(self.corpus.aspectSentences)
        print(self.corpus.aspectKeywords)
        for aspect in self.corpus.aspectKeywords:
            topPwords.append((aspect,[]))
        for word in self.corpus.wordFreq.keys():
            maxChi = 0.0  # max chi-sq value for this word
            maxAspect = ""  # corresponding aspect
            for aspect in self.corpus.aspectKeywords:
                print(aspect,word)
                self.aspectWordMat[aspect][word] = self.chiSq(aspect, word)
                if self.aspectWordMat[aspect][word] > maxChi:
                    maxChi = self.aspectWordMat[aspect][word]
                    maxAspect = aspect
            if maxAspect != "":
                topPwords[maxAspect].append((maxChi, word))

        changed = False
        for aspect in self.corpus.aspectKeywords:
            for t in heapq.nlargest(self.p, topPwords[aspect]):
                if t[1] not in self.corpus.aspectKeywords[aspect]:
                    changed = True
                    self.corpus.aspectKeywords[aspect].append(t[1])
        return changed

    # Populate wList,ratingsList and reviewIdList
    def populateLists(self):
        for review in self.corpus.allReviews:
            # Computing W matrix for each review
            W = defaultdict(lambda: defaultdict(int))
            for sentence in review.sentences:
                if len(sentence.assignedAspect) == 1:
                    for word, freq in sentence.wordFreqDict.items():
                        W[sentence.assignedAspect[0]][word] += freq
            if len(W) != 0:
                self.wList.append(W)
                self.ratingsList.append(review.ratings)
                self.reviewIdList.append(review.reviewId)

    def bootStrap(self):
        changed = True
        while self.iter > 0 and changed:
            self.iter -= 1
            print(self.corpus)
            self.corpus.aspectSentences.clear()
            for review in self.corpus.allReviews:
                for sentence in review.sentences:
                    self.assignAspect(sentence)
            self.populateAspectWordMat()
            changed = self.calcChiSq()
        self.corpus.aspectSentences.clear()
        for review in self.corpus.allReviews:
            for sentence in review.sentences:
                self.assignAspect(sentence)
        print(self.corpus.aspectKeywords)

    # Saves the object into the given file
    def saveToFile(self, fileName, obj):
        with open(fileName, 'w') as fp:
            json.dump(obj, fp)
            fp.close()
#
#
# rd = ReadData()
# rd.readAspectSeedWords()
# rd.readStopWords()
# rd.readReviewsFromJson()
# rd.removeLessFreqWords()

def ruunn(rd):
    bootstrapObj = BootStrap(rd)
    bootstrapObj.bootStrap()
    bootstrapObj.populateLists()
    bootstrapObj.saveToFile("wList.json", bootstrapObj.wList)
    bootstrapObj.saveToFile("ratingsList.json", bootstrapObj.ratingsList)
    bootstrapObj.saveToFile("reviewIdList.json", bootstrapObj.reviewIdList)
    bootstrapObj.saveToFile("vocab.json", list(bootstrapObj.corpus.wordFreq.keys()))
    bootstrapObj.saveToFile("aspectKeywords.json", bootstrapObj.corpus.aspectKeywords)