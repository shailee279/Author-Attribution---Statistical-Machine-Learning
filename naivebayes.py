#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:21:19 2019

@author: swapnilshailee
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/Users/swapnilshailee/Downloads/Who-Tweeted-That-master/enhanced_train_tweets.csv',names = ["userID","tweets"])
df.head()
df = df[pd.notnull(df['tweets'])]
df.info()
col = ['userID', 'tweets']
df = df[col]
df.columns
df.columns = ['userID', 'tweets']
df['category_id'] = df['userID'].factorize()[0]
category_id_df = df[['userID', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'userID']].values)
df.head()
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.tweets).toarray()
labels = df.category_id
features.shape
X_train, X_test, y_train, y_test = train_test_split(df['tweets'], df['userID'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("tfidf done")
dtc = DecisionTreeClassifier(criterion="entropy")
bag_model=BaggingClassifier(base_estimator=dtc, n_estimators=100, bootstrap=True)
bag_model=bag_model.fit(X_train_tfidf, y_train)
print("Training done")
import re
def clean_tweet_text(tweet):
    text = re.sub(r'@\w+\s?', '', tweet)
    text = re.sub(r'http.?://[^\s]+[\s]?', '', text)
#     text = re.sub(r'#\w+\s?', '', tweet)
    text = re.sub('[^a-zA-Z\s]\s?', '', text)
    text = text.lower()
    return text
print("cleaning done")
line_num = 1
with open('./whodunnit/test_tweets_unlabeled.txt') as un_fd:
    with open('result_bagging.txt','w') as res:
        for line in un_fd.readlines():
            clean_line = clean_tweet_text(line)
            userID = xgbval.predict(count_vect.transform([clean_line])).tolist()
            print (userID)
            res.write("%s\t%s\n"%(line_num,userID[0]))
            line_num += 1