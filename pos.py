#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:43:21 2019

@author: swapnilshailee
"""

import pandas as pd

dataset = pd.read_csv('/Users/swapnilshailee/Desktop/whodunnit/merged-data.csv')



print(dataset)
dataset = dataset.drop(['Emoji Count'], axis = 1) 
objtweet = dataset['tweet']
tweets = objtweet.values
"""
tweets = objtweet.values
emoji_list = []
emojiset= [':D',':)',';)',':(',':P','>.>','=/','= ]','= D','xD', '=(','=(',':O',';-)',':-(','*mwuah*','<3','*sigh*',':/','^sv','^dr']
for tweet in tweets:
    val = 0
    for word in tweet:
        for emoji in emojiset:
            if emoji in word.split():
                val += 1
    emoji_list.append(val)



ems = pd.DataFrame(emoji_list)
ems.columns = ['Emoji Count']
dataset = dataset.join(ems)
print("Emoji Count done")

upper_list = []
for tweet in tweets:
    val = 0
    for word in tweet:
        for w in word.split():
            if w.upper() == w:
                val += 1
    upper_list.append(val)

ups = pd.DataFrame(upper_list)
ups.columns = ['Capital Words Count']
dataset = dataset.join(ups)
print("Capital Count done")
"""
dataset.to_csv('merged-data.csv', encoding='utf-8', index=False)