#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:12:35 2021

@author: razzon
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


zomato_dataset = pd.read_csv("clean.csv")

zomato_dataset.dropna(inplace =True)

lko_rest = zomato_dataset

def recommend(cost,cuisine,location, lko_rest = lko_rest):
    
    cuisine = [cuisine]
    location = [location]
  
    lko_rest1 = lko_rest.copy().loc[lko_rest['location'] == location[0]]
    
    for i in range(1, len(location)):
        lko_rest2 = lko_rest.copy().loc[lko_rest['location'] == location[i]]
        lko_rest1 = pd.concat([lko_rest1, lko_rest2])
        lko_rest1.drop_duplicates(subset='name', keep='last', inplace=True)
    
    lko_rest_locale = lko_rest1.copy()
    
    lko_rest_locale = lko_rest_locale.loc[lko_rest_locale['cost'] <= cost]
    
    lko_rest_locale['Start'] = lko_rest_locale['cuisines'].str.find(cuisine[0])
    lko_rest_cui = lko_rest_locale.copy().loc[lko_rest_locale['Start'] >= 0]
    
    for i in range(1, len(cuisine)):
        lko_rest_locale['Start'] = lko_rest_locale['cuisines'].str.find(cuisine[i])
        lko_rest_cu = lko_rest_locale.copy().loc[lko_rest_locale['Start'] >= 0]
        lko_rest_cui = pd.concat([lko_rest_cui, lko_rest_cu])
        lko_rest_cui.drop_duplicates(subset='name', keep='last', inplace=True)
    
    lko_rest_cui = lko_rest_cui.sort_values('Mean Rating', ascending=False)
    lko_rest_cui.drop_duplicates(subset='name', keep='last', inplace=True)
    
    lko_rest1 = lko_rest_cui 
    
    lko_rest1 = lko_rest1.reset_index()

    if lko_rest1.empty: 
        return None
        
    else:
        
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(lko_rest1['reviews_list'])
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        cosine_sim2 = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        sim = list(enumerate(cosine_sim2[0]))
        sim = sorted(sim, key=lambda x: x[1], reverse=True)
        sim = sim[1:11]
        indi = [i[0] for i in sim]
        
        final = lko_rest1.copy().iloc[indi[0]]
        final = pd.DataFrame(final)
        final = final.T
        
        for i in range(1, len(indi)):
            final1 = lko_rest1.copy().iloc[indi[i]]
            final1 = pd.DataFrame(final1)
            final1 = final1.T
            final = pd.concat([final, final1])
        
        rest_sugg = final
        
        rest_list1 = rest_sugg.copy().loc[:,['name','cuisines','Mean Rating', 'cost','location']]
        # rest_list1.drop_duplicates(subset='name', keep='last', inplace=True)
        rest_list = pd.DataFrame(rest_list1).sort_values('Mean Rating', ascending=False)
        rest_list = rest_list.reset_index()
        rest_list = rest_list.rename(columns={'index': 'res_id'})
        rest_list.drop('res_id', axis=1, inplace=True)
        rest_list = rest_list.T
        rest_list = rest_list
        ans = rest_list.to_dict()
        res = [value for value in ans.values()]
        return res
       
    
recommend(600,'pizza','banashankari')
