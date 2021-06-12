#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Displaying the top 10 data in the dataset
zomato_dataset = zomato_dataset.head(100)

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
# zomato_dataset.drop(['phone', 'dish_liked', 'url'], axis=1, inplace=True)

# Checking for the distinct values in the dataset
zomato_dataset.nunique()

# Removing the duplicate calues from the dataset
zomato_dataset.drop_duplicates(inplace=True)

# Checking again after removing duplicates
zomato_dataset.nunique()

# Dopping the null and na values from the dataset
zomato_dataset.dropna(inplace=True)

# Renaming some of the columns to easier names
zomato_dataset = zomato_dataset.rename(
    columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})
zomato_dataset.name = zomato_dataset.name.apply(lambda x: x.title())

# Mapping the yes ot no to 0 and 1 for some columns
map_value = {'Yes': 1, 'No': 0}
zomato_dataset['online_order'] = zomato_dataset['online_order'].map(map_value)
zomato_dataset['book_table'] = zomato_dataset['book_table'].map(map_value)

'''
Some Transforamtions for certain columns
'''

# Changing the NEW, - and /5 values in rate column to numeric
zomato_dataset['rate'] = zomato_dataset['rate'].replace('NEW', '0')
zomato_dataset['rate'] = zomato_dataset['rate'].replace('-', '0')
zomato_dataset['rate'] = zomato_dataset['rate'].str.replace('/5', '').astype(float)

# Computing Mean Rating
restaurants = list(zomato_dataset['name'].unique())
zomato_dataset['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato_dataset['Mean Rating'][zomato_dataset['name'] == restaurants[i]] = zomato_dataset['rate'][
        zomato_dataset['name'] == restaurants[i]].mean()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(1, 5))
zomato_dataset[['Mean Rating']] = scaler.fit_transform(zomato_dataset[['Mean Rating']]).round(2)

# =============================================================================
# Text preprocessing
# =============================================================================

# Converting cost to integer
zomato_dataset['cost'].replace(',', "")
zomato_dataset['cost'] = pd.to_numeric(zomato_dataset['cost'])

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


# Modifying the review list byr removing the rated number
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


zomato_dataset['reviews_list'] = zomato_dataset['reviews_list'].apply(remove_numbers)

zomato_dataset['reviews_list'] = zomato_dataset['reviews_list'].str.replace('rated  ratedn', '')

# Dropping down the unnecessary column
# zomato_dataset = zomato_dataset.drop(
    # ['address', 'rest_type', 'type', 'menu_item', 'votes', 'online_order', 'book_table', 'city'], axis=1)


# Removing whitespaces
def remove_whitespace(text):
    return " ".join(text.split())


zomato_dataset['reviews_list'] = zomato_dataset['reviews_list'].apply(remove_whitespace)

zomato_dataset['location'] = zomato_dataset['location'].str.lower()

# =============================================================================
# Model Creation
# =============================================================================

# Creating tf-idf matrix

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(zomato_dataset['reviews_list'])

cos_sim = linear_kernel(X, X)


def recommend(name, cosine_similarities=cos_sim):
    recommend_restaurant = []

    idx = zomato_dataset[zomato_dataset == name].index[0]

    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)

    top10_indexes = list(score_series.iloc[0:11].index)

    for each in top10_indexes:
        recommend_restaurant.append(list(zomato_dataset.index)[each])

    df_new = pd.DataFrame(columns=['name', 'cuisines', 'Mean Rating', 'cost', 'location'])

    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(zomato_dataset[['name', 'cuisines', 'Mean Rating', 'cost', 'location']][zomato_dataset.index == each].sample()))

    df_new = df_new.drop_duplicates(subset=['name', 'cuisines', 'Mean Rating', 'cost', 'location'], keep=False)

    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)

    print('Top %s Resturents like %s : ' % (str(len(df_new)), name))

    return df_new
