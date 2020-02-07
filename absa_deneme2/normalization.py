import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from pattern.text.en import tag
import pandas as pd
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn

stopword_list = nltk.corpus.stopwords.words('english')
stopword_list += 'francisco','orleans','juan','puerto','rico','ritz','cambridge','milano','toronto','monaco','seattle','hotel','lot','\n','able','100','nyc'
wnl = WordNetLemmatizer()


def tokenize_sentences(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return word_tokens


def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens if  len(token)>2]
    a=[]
    puncs = set(string.punctuation)
    for word in tokens:
        if not all(c.isdigit() or c in puncs for c in word):
            word = word.lower()
            a.append(a)
    return tokens

def pos_tag_text(text):
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('JJ'):
            return wn.ADJ
        elif pos_tag.startswith('NN'):
            return wn.NOUN
        else:
            return None

    tagged_text = nltk.pos_tag(text)
    tagged_lower_text =  [(word.lower(), penn_to_wn_tags(pos_tag))
                             for word, pos_tag in
                             tagged_text if (pos_tag.startswith('JJ') or pos_tag.startswith('NN'))]
    return tagged_lower_text


def lemmatize_text(text):
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in
                                    tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in
                       stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def normalized_corpus(corpus,tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text=tokenize_text(text)
        text = lemmatize_text(text)
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        if(tokenize):
            text=tokenize_text(text)
        normalized_corpus.append(text)
    print(normalized_corpus)
    return normalized_corpus


def get_data(filename):
    data = pd.read_csv(filename, error_bad_lines=False)
    return data
