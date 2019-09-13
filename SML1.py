#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:27:43 2019

@author: swapnilshailee
"""

from nltk import tokenize
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

author_tweet_dict = {}

with open ('/Users/swapnilshailee/Desktop/whodunnit/train_tweets.txt') as fd:
    for n,line in enumerate(fd.readlines()):
        two_col = line.split('\t')
        if two_col[0] in author_tweet_dict.keys():
            author_tweet_dict[two_col[0]].append(two_col[1])
        else:
            author_tweet_dict[two_col[0]] = [two_col[1]]

tweet_dataset = pd.DataFrame(list(author_tweet_dict.items()), columns=['author', 'tweet'])

tweet_dataset.head()

tweet_dataset['tweet_count'] = tweet_dataset['tweet'].apply(len)
print("Tweet Count done")

objtweet = tweet_dataset['tweet']
tweets = objtweet.values
word_count = []
for tweet in tweets:
    val = 0
    for word in tweet:
        print(len(word.split()))
        val = val + len(word.split())
    word_count.append(val)
    
wordc = pd.DataFrame(word_count)

wordc.columns = ['Word Count']

final_dataset = tweet_dataset.join(wordc)
print("Word Count done")

final_dataset['tweet_density'] = final_dataset['Word Count'] / (final_dataset['tweet_count']+1)

print("Tweet density")
import re
objtweet = tweet_dataset['tweet']
tweets = objtweet.values
url_count = []
for tweet in tweets:
    val = 0
    for word in tweet:
        match = re.search(r'\w+@\w+', word)
        if match:
            val += 1
    url_count.append(val)


urls = pd.DataFrame(url_count)
urls.columns = ['Email/Website Count']
dataset = final_dataset.join(urls)
print("Website count done")
import re
objtweet = tweet_dataset['tweet']
tweets = objtweet.values
http_list = []
for tweet in tweets:
    val = 0
    print("TWEET")
    for word in tweet:
        if re.search("(?P<url>https?://[^\s]+)", word):
            val += 1
    http_list.append(val)
https = pd.DataFrame(http_list)
https.columns = ['HTTP Count']
dataset = dataset.join(https)
print("HTTP Count done")
objtweet = tweet_dataset['tweet']
tweets = objtweet.values
retweet_list = []
cmp = 'RT'
for tweet in tweets:
    val = 0
    for word in tweet:
        if cmp in word.split():
            val += 1
    retweet_list.append(val)



retweets = pd.DataFrame(retweet_list)
retweets.columns = ['Retweet Count']
dataset = dataset.join(retweets)
print("Retweet Count done")

tweets = objtweet.values
hashtag_list = []
for tweet in tweets:
    sum = 0
    for word in tweet:
        val = len(re.findall(r'(?i)\#\w+', word))
        sum += val
    hashtag_list.append(sum)



hashs = pd.DataFrame(hashtag_list)
hashs.columns = ['Hashtag Count']
dataset = dataset.join(hashs)
print("Hash tag done")

import spacy 
  
# Load English tokenizer, tagger, 
# parser, NER and word vectors 
nlp = spacy.load("en_core_web_sm")
print("spacy imported")
postagnounlist = []
postagdetlist = []
postagpunctlist = []
postagadplist = []
postagverblist = []
postagpropnlist = []
postagadjlist = []

for tweet in tweets:
    verb_val = 0
    det_val = 0
    noun_val = 0
    punct_val = 0
    propn_val = 0
    adp_val = 0
    adj_val = 0
    for word in tweet:
        doc = nlp(word)
        for token in doc:
            if token.pos_ == "VERB":
                verb_val += 1
            if token.pos_== "DET":
                det_val += 1
            if token.pos_ == "NOUN" :
                noun_val += 1
            if token.pos_ == "PUNCT" :
                punct_val += 1
            if token.pos_ == "ADP":
                adp_val += 1
            if token.pos_ == "PROPN":
                propn_val += 1
            if token.pos_ == "ADJ":
                adj_val += 1
    postagnounlist.append(noun_val)
    postagdetlist.append(det_val)
    postagpunctlist.append(punct_val)
    postagpropnlist.append(propn_val)
    postagverblist.append(verb_val) 
    postagadplist.append(adp_val)
    postagadjlist.append(adj_val)

print("lists created")               
nouns = pd.DataFrame(postagnounlist)
nouns.columns = ['Noun Count']
dataset = dataset.join(nouns)
print("NLP noun tag done")   
dets = pd.DataFrame(postagdetlist)
dets.columns = ['Det Count']
dataset = dataset.join(dets)
print("NLP Det tag done")            
verbs = pd.DataFrame(postagverblist)
verbs.columns = ['Verb Count']
dataset = dataset.join(verbs)
print("NLP verbs tag done")   
puncts = pd.DataFrame(postagpunctlist)
puncts.columns = ['Punctuation Count']
dataset = dataset.join(puncts)
print("NLP punctuation tag done") 
adps = pd.DataFrame(postagadplist)
adps.columns = ['ADP Count']
dataset = dataset.join(adps)
print("NLP adposition tag done") 
adjs = pd.DataFrame(postagadjlist)
adjs.columns = ['Adjective Count']
dataset = dataset.join(adjs)
print("NLP Adjective tag done") 
propns = pd.DataFrame(postagpropnlist)
propns.columns = ['Proper Noun Count']
dataset = dataset.join(propns)
print("NLP Proper Noun tag done") 
                


dataset.to_csv('merged-data.csv', encoding='utf-8', index=False)
