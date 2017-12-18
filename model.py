import json
import os
import pickle
import string

import numpy as np
from nltk import sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

COUNT = 4000
COUNT_TEST = 100

VECTORIZER_PATH = 'vectorizer.dump'


class Tokenizer:
    def __init__(self):
        self.stemmer = RussianStemmer("russian")
        self._punct = set(string.punctuation + "«»№-—")

    def __call__(self, text):
        tokens = []

        for sent in sent_tokenize(text):
            for word in wordpunct_tokenize(sent):
                # skip trash
                if all(char in self._punct for char in word):
                    continue

                tokens.append(self.stemmer.stem(word))

        return tokens


def filter_data(data):
    return [entry for entry in data if entry['rating'] is not None]


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    l2_loss = np.mean(np.abs(y_test - y_pred) ** 2)
    print('L2 loss: {}'.format(l2_loss))


if __name__ == '__main__':
    with open('data.json', 'r') as f:
        data = json.load(f)

    data = filter_data(data)[:COUNT]
    corpus = [entry['text'] for entry in data]

    if os.path.exists(VECTORIZER_PATH):
        print('Loading vectorizer...')
        vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    else:
        print('Vectorizing...')
        vectorizer = CountVectorizer(stop_words=stopwords.words("russian"), tokenizer=Tokenizer())
        vectorizer.fit(corpus)

        print('Saving vectorizer...')
        pickle.dump(vectorizer, open(VECTORIZER_PATH, 'wb'))

    print('Transforming data...')
    X = vectorizer.transform(corpus).toarray()
    y = np.array([entry['rating'] for entry in data])

    X_train, X_test = X[:COUNT - COUNT_TEST], X[COUNT - COUNT_TEST:]
    y_train, y_test = y[:COUNT - COUNT_TEST], y[COUNT - COUNT_TEST:]

    print('Training...')
    model = train_model(X_train, y_train)

    print('\nTrain')
    test_model(model, X_train, y_train)

    print('\nTest')
    test_model(model, X_test, y_test)

    print('Saving model...')
    pickle.dump(model, open('model.dump', 'wb'))
