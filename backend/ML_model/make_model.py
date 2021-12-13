# Standard Libs
import pickle
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
from collections import Counter
#from PIL import Image

from custom_transformers import *
#from viz_utils import *
from ml_utils import *

# ML libs
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import shap
from main import *

# Creating an instance for the homemade class BinaryClassifiersAnalysis
clf_tool = BinaryClassifiersAnalysis()
clf_tool.fit(set_classifiers, X_train_prep, y_train, random_search=True, cv=5, verbose=5)

# 5

new_restaurants_prep = full_pipeline.fit_transform(new_restaurants.drop('target', axis=1))
print(new_restaurants_prep.shape)

model = clf_tool.classifiers_info['LightGBM']['estimator']
y_pred = model.predict(new_restaurants_prep)
y_probas = model.predict_proba(new_restaurants_prep)
y_scores = y_probas[:, 1]

# Labelling new data
new_restaurants['success_class'] = y_pred
new_restaurants['success_proba'] = y_scores

pickle.dump(model, open('ml_model.pkl', 'wb'))

# JUGAAD
'''
dp = r'Test.csv'
rd = pd.read_csv('Dataset/Test.csv')

rd2 = pd.read_csv('Dataset/zomato.csv')
rd3 = rd2.append(rd, ignore_index = True)

tr1, nr1 = common_pipeline.fit_transform(rd3)

nr_prep = full_pipeline.fit_transform(nr1.drop('target', axis=1))
print(nr_prep.shape)

y_pd = model.predict(nr_prep)
y_pb = model.predict_proba(nr_prep)
y_sc = y_pb[:, 1]

# Labelling new data
nr1['success_class'] = y_pd
nr1['success_proba'] = y_sc
nr1['success_proba'].iloc[-1]

nrd = nr1.reset_index().merge(rd3.reset_index()[['name', 'index']], how='left', on='index')
#nrd.iloc[-1]
'''
