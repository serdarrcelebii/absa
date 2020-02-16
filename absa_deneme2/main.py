from preprocess import *
from similarity import *
from normalization import *
from topic_modeling import *
from plsa import  *
from segmentation import *

if __name__ == "__main__":
    documentnotprocess=get_data('test.csv')
    #documents=processed_data()
    #print(documentnotprocess)
    topic=5
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

    #print_coherence(tokenize_corpus)
    print("-----------------------lsi-----------------------------")
    lsi=train_lsi_model_gensim(tokenize_corpus,topic)
    print_topics_gensim(lsi,topic)
    print("-----------------------lda-----------------------------")
    lda = train_lda_model_gensim(tokenize_corpus, topic)
    print_topics_gensim(lda, topic)
    print("-----------------------nmf-----------------------------")
    train_nmf_model(nottokenize_corpus,topic)
    #topicwords=dotopicanalysis(nottokenize_corpus,stopword_list,topic)
    aspect_list=[]
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


