import nltk
import re
import string
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd

ps = LancasterStemmer()
wnl = WordNetLemmatizer()


def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(remove_characters_before_tokenization(sentence)) for sentence in sentences]
    return word_tokens


def stemmer(tokens):
    stem = [ps.stem(token.lower()) for token in tokens]
    return stem;


def lemmatize(tokens):
    stem = [wnl.lemmatize(token.lower()) for token in tokens]
    return stem;


def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens


def remove_characters_before_tokenization(sentence,
                                          keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]'  # add other characters here to remove them
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    else:
        PATTERN = r'[^a-zA-Z ]'  # only extract alpha-numeric characters
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence


# def expand_contractions(sentence, contraction_mapping):
#    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
#                                     flags=re.IGNORECASE | re.DOTALL)

# def expand_match(contraction):
#   match = contraction.group(0)
#   first_char = match[0]
#   expanded_contraction = contraction_mapping.get(match) \
#       if contraction_mapping.get(match) \
#       else contraction_mapping.get(match.lower())
##   expanded_contraction = first_char + expanded_contraction[1:]
#  return expanded_contraction

# expanded_sentence = contractions_pattern.sub(expand_match, sentence)
# return expanded_sentence


def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


def get_data(filename):
    data = pd.read_csv(filename, error_bad_lines=False)
    return data


def data_operations(data):
    datalist = []
    for item in data['Review']:
        tokenize = tokenize_text(item)
        tokens = tokenize[0]
        tokens = remove_repeated_characters(remove_characters_after_tokenization(tokens))
        # tokens=stemmer(tokens)
        tokens = lemmatize(tokens)
        tokens = remove_stopwords(tokens)
        datalist.append(tokens)
    return datalist


def processed_data():
    data = get_data('test.csv')
    return data_operations(data)


def normalize_corpus(corpus, lemmatizse=True, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        if lemmatizse:
            text = lemmatize(text)
        else:
            text = text.lower()
        text = remove_characters_after_tokenization(text)
        text = remove_stopwords(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)
        return normalized_corpus
