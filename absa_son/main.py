from preprocess import *
from normalization import *
from topic_modeling import *
from BootStrap import *
from ReadData import *
from Review import *

if __name__ == "__main__":
    documentnotprocess=get_data('dataset.csv')
    #documents=processed_data()
    #print(documentnotprocess)
    topic=4
    tokenize_corpus=normalized_corpus( documentnotprocess['Review'],True)
    nottokenize_corpus=normalized_corpus( documentnotprocess['Review'])
    print(tokenize_corpus)
    print(nottokenize_corpus)
    l=[]
    ta=[]
    index=0
    sno=documentnotprocess['S.No.'][0:].to_numpy()
    ratings=documentnotprocess['Rating'][0:].to_numpy()

    for t in tokenize_corpus:
        ta.append(t)
        index=index+1
    reviews=pd.DataFrame(list(zip(sno, ta, ratings)),columns =['S.No.', 'Review', 'Rating'])

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 4febebdf643f9462c835f82481f1b2c004a30459
    # print_coherence(tokenize_corpus)
    # print("-----------------------lsi-----------------------------")
    # lsi=train_lsi_model_gensim(tokenize_corpus,topic)
    # print_topics_gensim(lsi,topic)
    # print("-----------------------lda-----------------------------")
    # lda = train_lda_model_gensim(tokenize_corpus, topic)
    # print_topics_gensim(lda, topic)
    # print("-----------------------nmf-----------------------------")
    # train_nmf_model(nottokenize_corpus,topic)
<<<<<<< HEAD
    #topicwords=dotopicanalysis(nottokenize_corpus,stopword_list,topic)
=======
=======
<<<<<<< HEAD
    #print_coherence(tokenize_corpus)
=======
    print_coherence(tokenize_corpus)
>>>>>>> 3a185f6c65b26f67ee0fad06eca135619788a0da
    print("-----------------------lsi-----------------------------")
    lsi=train_lsi_model_gensim(tokenize_corpus,topic)
    print_topics_gensim(lsi,topic)
    print("-----------------------lda-----------------------------")
    lda = train_lda_model_gensim(tokenize_corpus, topic)
    print_topics_gensim(lda, topic)
    print("-----------------------nmf-----------------------------")
    train_nmf_model(nottokenize_corpus,topic)
<<<<<<< HEAD
    # topicwords=dotopicanalysis(nottokenize_corpus,stopword_list,topic)1
=======
>>>>>>> 1f5300411ab5d1a256843a0a21af447a9f6c1ddf
    #topicwords=dotopicanalysis(nottokenize_corpus,stopword_list,topic)1
>>>>>>> 3a185f6c65b26f67ee0fad06eca135619788a0da
>>>>>>> 4febebdf643f9462c835f82481f1b2c004a30459
    # aspect_list
    # file = codecs.open('topics.txt', 'r', 'utf-8')
    # for line in file:
    #     print(line)
    #     for i in line.split(' '):
    #         if i not in aspect_list:
    #             aspect_list.append(i)
    #             for a in tokenize_corpus:
    #                 for j in a:
    #                     print(j)
    #                     if i!=j and len(i)>1 and len(j)>1:
    #                        para = paradigSimilarity(i, j)
    #                        print(para)
    #                        if para > 0.075:
    #                            if j not in aspect_list:
    #                                 aspect_list.append(j)
    # print(aspect_list)
    rd = ReadData()
    rd.readAspectSeedWords()
    rd.readReviewsFromJson(reviews)
    rd.removeLessFreqWords()
    print(rd.aspectKeywords)
    ruunn(rd)
    # rd = ReadData()
    # rd.readAspectSeedWords()
    # rd.readReviewsFromJson(reviews)
    # rd.removeLessFreqWords()
    # print(rd.aspectKeywords)
    # ruunn(rd)

    # filew = codecs.open('topics.txt', 'r', 'utf-8')
    # for line in file:
    #     print(line)
    #     for i in line.split(' '):
    #         if len(i)>0:
    #             print(i)
    #             for linew in filew:
    #                 print(linew)
    #                 for j in linew.split(' '):
    #                     if i!=j and len(j)>0:
    #                         print(j)
    #                         print(paradigSimilarity(i,j))
    #file.close()


