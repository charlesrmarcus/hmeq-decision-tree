import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.base import TransformerMixin

initial_data = pd.read_csv("../hmeq-data/hmeq.csv")
manipulated_data = pd.read_csv("../hmeq-data/hmeq.csv")

# print manipulated_data.head()

# Impute Mean Values for NaN Data
# Cited from source: https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values. Columns of dtype object are imputed with the most
        frequent value in the column. Columns of other types are imputed with mean of column

        Columns of other types are imputed with mean of column."""
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
                              if X[c].dtype == np.dtype('O')
                              else X[c].mean() for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

X = pd.DataFrame(manipulated_data)
transformed_X = DataFrameImputer().fit_transform(X)

#print ('before...')
#print (X)
#print ('after...')
#print (transformed_X)

# Nominal Features will be added back later once reformatted - Kaggle Citation
hmeq_features = transformed_X.drop(columns=["BAD", "JOB", "REASON"])
hmeq_target = transformed_X["BAD"]

#print hmeq_features

# Set up Initial NN Structure
clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100, 100), max_iter=5000, alpha=1e-5)
train, test, train_labels, test_labels = train_test_split(hmeq_features, hmeq_target, test_size=0.33, random_state=42)

# Cross Validation
scores = cross_val_score(clf, hmeq_features, hmeq_target, cv=5)
print "Cross Validation Scores: ", scores
