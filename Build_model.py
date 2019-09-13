#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
import re


# In[2]:


df = pd.read_csv('train_tweets_full.csv',names = ["userID","tweets"])
df = df[pd.notnull(df['tweets'])]
df.info()


# In[3]:


col = ['userID', 'tweets']
df = df[col]
df.columns = ['userID', 'tweets']
df['category_id'] = df['userID'].factorize()[0]
category_id_df = df[['userID', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'userID']].values)
df.head()


# In[4]:


#TF-IDF Settings
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')


# In[5]:


X_train = df['tweets']
y_train = df['userID']
# tfidf.fit(df['tweets'])


# In[6]:


xtrain_tfidf =  tfidf.fit_transform(X_train)
count_vect = CountVectorizer(analyzer='word', stop_words='english')
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(xtrain_tfidf)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[ ]:


# #Liner Model Logistic Regression
# clf = LogisticRegression().fit(X_train_tfidf, y_train)
# joblib.dump(model, 'LogisticReg_tfidf_2950.pkl')

# In[ ]:

# MLP Classifier

clf = MLPClassifier(activation='relu', solver='sgd', hidden_layer_sizes=(3,5), random_state=1)
clf.fit(x_train_tfidf,y_train)
def clean_tweet_text(tweet):
    text = re.sub(r'@\w+\s?', '', tweet)
    text = re.sub(r'http.?://[^\s]+[\s]?', '', text)
    text = re.sub('#\w+\s?', '', text)
    text = text.lower()
    return text

line_num = 1
with open('./whodunnit/test_tweets_unlabeled.txt') as un_fd:
    with open('result_mlp_tfidf_full.txt','w') as res:
        for line in un_fd.readlines():
            clean_line = clean_tweet_text(line)
            userID = clf.predict(count_vect.transform([clean_line])).tolist()
            print (userID)
            res.write("%s\t%s\n"%(line_num,userID[0]))
            line_num += 1
# In[ ]:


print(clf.predict(count_vect.transform(["RT @handle: Director of Global Brand Marketing, Hotels and Casino's $125k + 30% bonus - Orlando Fl http://bit.ly/4kUmBB #jobs #twitjobs"])))


# In[ ]:




