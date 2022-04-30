#Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


#Functionalities
from collections import Counter
import sys, os
import warnings
warnings.filterwarnings('ignore')

#NLP
import string
import re
import nltk


# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Custom Transformer
# from src.preprocess.preprocessor import TextPreprocessor

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier

import time

from catboost import CatBoostClassifier
from ..preprocess.preprocessor import TextPreprocessor
from ..models.metrics import evaluate_model


def train_model(clf, xtrain, xtest, ytrain, ytest, list):
    print(f'Classifier: {clf}')
    start_time = time.time()

    model = Pipeline(steps=[
        ("preprocessor", TextPreprocessor()),
        ("vectorizer", TfidfVectorizer()),
        ("clf", clf)])
        
    # Fit And Predict Model
    model = model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    try:
        ypred_proba = model.predict_proba(xtest)
    except:
        print('Predict_Proba not available')

    pickle.dump(model, open('../models/' + clf.__class__.__name__ + '.pkl', 'wb'))
    time_taken = time.time() - start_time
    
    print(f'Execution time: {round(time_taken, 2)}s')

    try:
        evaluate_model(ytest, ypred, ypred_proba)
    except:
        evaluate_model(ytest, ypred)
    
    list.append([clf.__class__.__name__,
    f'{round(accuracy_score(ytest, ypred), 2)*100}%',
    f'{int(round(time_taken, 0))}'])
    print('------------------------------------------------------')



def optimize_model(clf, param_grid, scoring, cv, xtrain, xtest, ytrain, ytest):
    start_time = time.time()

    pipe = Pipeline(steps=[
        ("preprocessor", TextPreprocessor()),
        ("vectorizer", TfidfVectorizer()),
        ("clf", clf)])
    
    model = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, cv=cv, scoring=scoring)

    # Fit And Predict Model
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    try:
        ypred_proba = model.predict_proba(xtest)
    except:
        print('Predict_Proba not available')
    
    pickle.dump(model, open('../models/' + clf.__class__.__name__ + '_optimized.pkl', 'wb'))
    time_taken = time.time() - start_time
    
    print(f'Execution time: {round(time_taken, 2)}s')

    try:
        evaluate_model(ytest, ypred, ypred_proba)
    except:
        evaluate_model(ytest, ypred)

    print('------------------------------------------------------')