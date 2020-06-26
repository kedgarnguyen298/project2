import nltk, re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

# Preprocess Text
stop_words = stopwords.words('english')
normalizer = WordNetLemmatizer()

def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)
  pos_counts = Counter()
  pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )
  pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )
  pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )
  pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )
  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
  return most_likely_part_of_speech

def preprocess_text(text):
  cleaned = re.sub(r'\W+', ' ', text).lower()
  tokenized = word_tokenize(cleaned)
  normalized = " ".join([normalizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized])
  return normalized

# Load data
news_df = pd.read_csv('Articles.csv',encoding='latin-1')
user_df = pd.read_csv('user_read_set1.csv',encoding='latin-1')

news_df['Date'] =  pd.to_datetime(news_df['Date'].str.strip(), format='%m/%d/%Y')
user_df['Date'] =  pd.to_datetime(user_df['Date'].str.strip(), format='%m/%d/%Y')

user_df = user_df[user_df['Date'] >= pd.Timestamp(2016,1,1)] #get user recent news

news_df_temp = news_df.Heading.apply(preprocess_text)

# Tf-idf model for news headline and category
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.metrics import pairwise_distances

category_onehot_encoded = OneHotEncoder().fit_transform(np.array(news_df["NewsType"]).reshape(-1,1))
category_user_onehot_encoded = OneHotEncoder().fit_transform(np.array(user_df["NewsType"]).reshape(-1,1))

tfidf_headline_vectorizer = TfidfVectorizer(min_df = 0)
tfidf_headline_features = tfidf_headline_vectorizer.fit_transform(news_df_temp)

tfidf_user_headline_features = tfidf_headline_vectorizer.transform(user_df_temp)

def tfidf_based_model():
    for i in range (user_df.shape[0]):
        couple_dist = cosine_distances(tfidf_headline_features,tfidf_user_headline_features[i])
        category_dist = cosine_similarity(category_onehot_encoded, category_user_onehot_encoded[i])
        indices = np.argsort(couple_dist.ravel())[0:user_df.shape[0]]
        df = pd.DataFrame({
               'headline':news_df['Heading'][indices].values,
                'Cosine Distance with the queried article': couple_dist[indices].ravel(),
                'Category based Cosine Distance': category_dist[indices].ravel(), 
                'Category': news_df['NewsType'][indices].values,
                'Date': news_df['Date'][indices].values
        }).sort_values("Date",ascending=False).dropna()
        
        return df.iloc[1:,].head(10).style.hide_index()

tfidf_based_model()

# KNN model for news headline
from sklearn.neighbors import NearestNeighbors

n_neighbors = 20
KNN = NearestNeighbors(n_neighbors, p=2, metric='cosine')
KNN.fit(tfidf_headline_features)
NNs = KNN.kneighbors(tfidf_user_headline_features,return_distance=True)

def get_recommendation(top,news_df,scores):
    recommendation = pd.DataFrame(columns=['Heading','NewsType','Date','score'])
    count = 0
    for i in top:
        try:
            recommendation.at[count, 'Heading'] = news_df['Heading'][i]
            recommendation.at[count, 'NewsType'] = news_df['NewsType'][i]
            recommendation.at[count, 'Date'] = news_df['Date'][i]
            recommendation.at[count, 'score'] = scores[count]
            count += 1
        except KeyError:
            continue
    return recommendation 

top = NNs[1][0][1:]
index_scores = NNs[0][0][1:]

get_recommendation(top,news_df,index_scores)