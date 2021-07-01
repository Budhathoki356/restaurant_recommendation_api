#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:13:01 2021

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

# Reading the zomato dataset
zomato_dataset = pd.read_csv("zomato.csv")

# Selecting random 20000 data
zomato_dataset = zomato_dataset.sample(n = 30/home/budhathoki/PycharmProjects/Recommendation_api/rec100)

# getting the shape the dataset
zomato_dataset.shape

# Checking the descriptive statistics of the dataset
zomato_dataset.describe()


# =============================================================================
# Data Cleaning
# =============================================================================

# Checking the null values
zomato_dataset.isnull().sum()

# Discarding unnecessary columns from the dataset
zomato_dataset.drop(['phone','dish_liked' ,'url'], axis=1, inplace = True)
zomato_dataset = zomato_dataset.drop(['address','rest_type', 'listed_in(type)', 'menu_item', 'votes','online_order','book_table','listed_in(city)'],axis=1)


# Checking for the distinct values in the dataset
zomato_dataset.nunique()

# Removing the duplicate calues from the dataset
zomato_dataset.drop_duplicates(inplace = True)

# Checking again after removing duplicates
zomato_dataset.nunique()

# Dopping the null and na values from the dataset
zomato_dataset.dropna(inplace = True) 

# Renaming some of the columns to easier names
zomato_dataset = zomato_dataset.rename(columns={'approx_cost(for two people)':'cost'})
zomato_dataset.name = zomato_dataset.name.apply(lambda x:x.title())


'''
Some Transforamtions for certain columns
'''

# Changing the NEW, - and /5 values in rate column to numeric 
zomato_dataset['rate'] = zomato_dataset['rate'].replace('NEW','0')
zomato_dataset['rate'] = zomato_dataset['rate'].replace('-','0')
zomato_dataset['rate'] = zomato_dataset['rate'].str.replace('/5','').astype(float)


# Computing Mean Rating 
restaurants = list(zomato_dataset['name'].unique())
zomato_dataset['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato_dataset['Mean Rating'][zomato_dataset['name'] == restaurants[i]] = zomato_dataset['rate'][zomato_dataset['name'] == restaurants[i]].mean()

    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (1,5))
zomato_dataset[['Mean Rating']] = scaler.fit_transform(zomato_dataset[['Mean Rating']]).round(2)

# =============================================================================
# Text preprocessing
# =============================================================================

#Converting cost to integer
zomato_dataset['cost'] = zomato_dataset['cost'].str.replace(',','')
zomato_dataset['cost'] = pd.to_numeric(zomato_dataset['cost'])

zomato_dataset['location'] = zomato_dataset['location'].str.strip()

# Lower Casing the review list 
zomato_dataset["reviews_list"] = zomato_dataset["reviews_list"].str.lower()


# Removing the unnecessary punctuation 
import string
PUNCT_TO_REMOVE = string.punctuation

# Creating a user defined function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato_dataset["reviews_list"] = zomato_dataset["reviews_list"].apply(lambda text: remove_punctuation(text))


# Removing stopwords from the dataset
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# Creating a userdefined function to remove stopwords
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

zomato_dataset["reviews_list"] = zomato_dataset["reviews_list"].apply(lambda text: remove_stopwords(text))


# Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato_dataset["reviews_list"] = zomato_dataset["reviews_list"].apply(lambda text: remove_urls(text))


# Modifying the review list by removing the rated number
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

zomato_dataset['reviews_list']=zomato_dataset['reviews_list'].apply(remove_numbers)

zomato_dataset['reviews_list']=zomato_dataset['reviews_list'].str.replace('rated  ratedn','')

# Removing whitespaces
def remove_whitespace(text):
    return  " ".join(text.split())
  
zomato_dataset['reviews_list'] = zomato_dataset['reviews_list'].apply(remove_whitespace)

zomato_dataset['location'] = zomato_dataset['location'].str.strip()

zomato_dataset['location'] = zomato_dataset['location'].str.lower()

zomato_dataset['cuisines'] = zomato_dataset['cuisines'].str.lower()

lko_rest = zomato_dataset

def calc(cost,cuisine,location):
    
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
        
calc(800,'pizza','banashankari')