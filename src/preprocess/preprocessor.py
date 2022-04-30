import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import string
import re
import nltk
from nltk.util import ngrams
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import spacy
from spacy import displacy
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin



# Clean text

# strip spaces
def strip_spaces(text):
    return text.strip()

# remove uppercase letters
def remove_upper(text):
    return text.lower()

# clean html residue
def clean_html(text):
    text = re.sub(r'<.*?>', '', text)
    return text

# clean urls
def clean_url(text):
    text = re.sub(r'http\S+', 'url', text)
    return text

# clean newlines
def clean_newline(text):
    text = text.replace('\n', ' ')
    return text

# clean punctuation
def clean_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# clean numbers
def clean_numbers(text):
    text = re.sub(r'\d+', '', text)
    return text

# clean quotes 
def clean_quotes(text):
    return text.replace('’', '').replace('“', '').replace('”','').replace('‘','').replace(' — ',' ')

# remove stopwords
def clean_stopwords(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

# lemmatize text
def lemmatize(text):
    lemmatizer = nltk.WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word,pos='v') for word in text.split()]
    return text

# preprocessing function
def preprocessing(text): 
    text = strip_spaces(text)
    text = remove_upper(text)
    text = clean_html(text)
    text = clean_url(text)
    text = clean_newline(text)
    text = clean_punctuation(text)
    text = clean_numbers(text)
    text = clean_quotes(text)
    text = lemmatize(text)
    text = clean_stopwords(text)

    return text

class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.map(lambda x : preprocessing(x))
        return X