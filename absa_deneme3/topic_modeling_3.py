import csv

reviews = [row for row in csv.reader(open('test.csv'))]
print(reviews)

import re
import nltk

# We need this dataset in order to use the tokenizer
nltk.download('punkt')

from nltk.tokenize import word_tokenize

# Also download the list of stopwords to filter out
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def process_text(text):
    # Make all the strings lowercase and remove non alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text.lower())

    # Tokenize the text; this is, separate every sentence into a list of words
    # Since the text is already split into sentences you don't have to call sent_tokenize
    tokenized_text = word_tokenize(text)

    # Remove the stopwords and stem each word to its root
    clean_text = [
        stemmer.stem(word) for word in tokenized_text
        if word not in stopwords.words('english')
    ]

    # Remember, this final output is a list of words
    return clean_text


# Remove the first row, since it only has the labels
reviews = reviews[1:]

texts = [row[1] for row in reviews]
topics = [row[2] for row in reviews]

# Process the texts to so they are ready for training
# But transform the list of words back to string format to feed it to sklearn
texts = [" ".join(process_text(text)) for text in texts]

print(texts)



from gensim import corpora
texts = [process_text(text) for text in texts]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


from gensim import models
model = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

topics = model.print_topics(num_words=10)
for topic in topics:
    print(topic)


