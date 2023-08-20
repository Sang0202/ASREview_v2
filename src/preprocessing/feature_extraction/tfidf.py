import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# import joblib

# create a feature extraction model to transform the text data into a matrix of TF-IDF features

# create a function to fit the model to the training data 
# input: text data, number of features, ngram range
# output: fitted model

def fit_tfidf(texts, max_features=4, ngram_range=(1,1)):
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf.fit(texts)
    return tfidf

# create a function to transform the text data into a matrix of TF-IDF features
# input: fitted model, text data
# output: matrix of TF-IDF features

def transform_tfidf(tfidf, texts):
    tfidf_matrix = tfidf.transform(texts)
    return tfidf_matrix

# # create a function to save the fitted model with joblib
# # input: fitted model, model name
# # output: saved model

# def save_model(model, model_name):
#     joblib.dump(model, model_name)

# # create a function to load the fitted model with joblib
# # input: model name
# # output: loaded model

# def load_model(model_name):
#     model = joblib.load(model_name)
#     return model