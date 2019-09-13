#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, string


# In[ ]:


data = open('./whodunnit/train_tweets.txt','rb').read()
author, tweet = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split('\t')
    author.append(content[0])
    tweet.append(" ".join(content[1:]))


# In[ ]:





# In[ ]:


trainDF = pandas.DataFrame()
trainDF['author'] = author
trainDF['tweet'] = tweet


# In[ ]:


print (author)


# In[ ]:


author_count = len(set(author))
print (author_count)


# In[ ]:


# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['author'], trainDF['tweet'])


# In[ ]:


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


# In[ ]:


# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['tweet'])


# In[ ]:


# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


# In[ ]:


print (xtrain_count,xvalid_count)


# In[ ]:


tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
tfidf_vect.fit(trainDF['tweet'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)


# In[ ]:


print (xtrain_tfidf)


# In[ ]:


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# In[ ]:





# In[ ]:


# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("NB, WordLevel TF-IDF: ", accuracy)


# In[ ]:




