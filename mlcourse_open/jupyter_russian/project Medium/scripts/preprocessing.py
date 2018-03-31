# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from loaders import load_contents
from tools import translate_dataframe_to_english

# load_urls(os.path.join('data/', 'train.json'))

def variant_1_preprocessing():
    '''
    TfidfVectorizer

    '''
    # contents_train = load_contents(os.path.join('data/', 'train.json'))
    # contents_test = load_contents(os.path.join('data/', 'test.json'))
    # contents_train_translated = translate_dataframe_to_english(contents_train)
    # contents_test_translated = translate_dataframe_to_english(contents_test)

    contents_train_translated = list(pd.read_csv('persists/contents_train_translated.csv')['content'])
    contents_test_translated = list(pd.read_csv('persists/contents_test_translated.csv')['content'])

    tfidf_vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 2))
    X_train = tfidf_vectorizer.fit_transform(contents_train_translated)
    X_test = tfidf_vectorizer.transform(contents_test_translated)

    return X_train, X_test

































