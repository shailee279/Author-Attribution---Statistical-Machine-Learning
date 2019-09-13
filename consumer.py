#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

df = pd.read_csv('enhanced_train_tweets_2950.csv',names = ["userID","tweets"])
df.head()


# In[2]:


df = df[pd.notnull(df['tweets'])]


# In[3]:


df.info()


# In[4]:


col = ['userID', 'tweets']
df = df[col]


# In[5]:


df.columns


# In[6]:


# 'Product' = 'userID', 'Consumer_complaint_narrative' = 'tweets'
df.columns = ['userID', 'tweets']


# In[7]:


df['category_id'] = df['userID'].factorize()[0]
category_id_df = df[['userID', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'userID']].values)


# In[8]:


df.head()


# In[ ]:





# In[9]:


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.tweets).toarray()
labels = df.category_id
features.shape
print (features.shape)


# In[ ]:

check = 0
N = 2
for Product, category_id in sorted(category_to_id.items()):
  print ("check",check,"catid",category_id)
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  check += 1

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df['tweets'], df['userID'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# clf = MultinomialNB().fit(X_train_tfidf, y_train)
clf = LinearSVC().fit(X_train_tfidf, y_train)
joblib.dump(model, 'LinearSVC_chi2.pkl')

# In[ ]:


print(clf.predict(count_vect.transform(["RT @handle: Director of Global Brand Marketing, Hotels and Casino's $125k + 30% bonus - Orlando Fl http://bit.ly/4kUmBB #jobs #twitjobs"])))

