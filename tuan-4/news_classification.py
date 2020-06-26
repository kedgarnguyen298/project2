import pandas as pd
import numpy as np
import re
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import seaborn as sb

# Dataset preparation
def clean_text(text):
    
    #remove punctuation
    text = [char for char in text if char not in string.punctuation] 
    text_join = ''.join(text)
    
    #remove stopwords
    text_join_clean = [word for word in text_join.split() if word.lower() not in stopwords.words('english')] 
    
    #shorten word to their stem
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text_join_clean]
    text_join_clean = " ".join(stemmed_words)

    
    #return
    return text_join_clean

# Import data
news_df = pd.read_csv('Articles.csv',encoding='latin-1')
X = news_df.Article.apply(clean_text)
y = news_df.NewsType

#Split data
import sklearn.model_selection as model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33,random_state=42)

import pickle

pickle.dump(X_train, open('X_train.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))

pickle.dump(X_test, open('X_test.pkl', 'wb'))
pickle.dump(y_test, open('y_test.pkl', 'wb'))

import pickle

X_train = pickle.load(open('X_train.pkl', 'rb'))
y_train = pickle.load(open('y_train.pkl', 'rb'))

X_test = pickle.load(open('X_test.pkl', 'rb'))
y_test = pickle.load(open('y_test.pkl', 'rb'))

# Feature Engineering

#Count Vector as features
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(X_train)

# transform the training and validation data using count vectorizer object
X_train_count = count_vect.transform(X_train)
X_test_count = count_vect.transform(X_test)

#Tf-Idf Vectors as Features

# word level - we choose max number of words equal to 1000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=1000)
tfidf_vect.fit(X_train) # learn vocabulary and idf from training set

X_train_tfidf =  tfidf_vect.transform(X_train)

# assume that we don't have test set before
X_test_tfidf =  tfidf_vect.transform(X_test)

#Label Encoder
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
y_train_n = encoder.fit_transform(y_train)
y_test_n = encoder.fit_transform(y_test)
encoder.classes_

# BUILD MODEL
from sklearn import naive_bayes,metrics,svm
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train_count,y_train)
y_predict_train = clf.predict(X_train_count)
y_predict_test = clf.predict(X_test_count)
#y_predict_test

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(accuracy_score( y_test,y_predict_test))
print(classification_report(y_test,y_predict_test))

cfs = confusion_matrix(y_test,y_predict_test)
# True Positives
TP = cfs[1, 1]
print('True_Positive:' ,TP)
# True Negatives
TN = cfs[0, 0]
print('True_Negative:' ,TN)
# False Positives
FP = cfs[0, 1]
print('False_Positive:' ,FP)
# False Negatives
FN = cfs[1, 0]
print('False_Negative:' ,FN)

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train_count,y_train)
y_pred = model.predict(X_test_count)
#y_pred

print(classification_report(y_test,y_pred))
print(accuracy_score( y_test, y_pred))

cfs = confusion_matrix(y_test,y_pred)
# True Positives
TP = cfs[1, 1]
print('True_Positive:' ,TP)
# True Negatives
TN = cfs[0, 0]
print('True_Negative:' ,TN)
# False Positives
FP = cfs[0, 1]
print('False_Positive:' ,FP)
# False Negatives
FN = cfs[1, 0]
print('False_Negative:' ,FN)