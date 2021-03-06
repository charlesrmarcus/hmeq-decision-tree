import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import itertools
from sklearn import tree
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score

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
#print hmeq_target

train, test, train_labels, test_labels = train_test_split(hmeq_features, hmeq_target, test_size=0.33, random_state=42)
#print "Training Data", train, "Training Labels", train_labels

# Create Initial Classifier - Decision Tree
clf = tree.DecisionTreeClassifier()
fitted_clf = clf.fit(train, train_labels)

# Cross Validation
scores = cross_val_score(fitted_clf, hmeq_features, hmeq_target, cv=5)
print "Cross Validation Scores: ", scores

# Accuracy and Precision Scores
training_prediction = fitted_clf.predict(train)
training_accuracy = accuracy_score(training_prediction, train_labels) * 100
print "Training Accuracy: ", training_accuracy

testing_prediction = fitted_clf.predict(test)
testing_accuracy = accuracy_score(testing_prediction, test_labels) * 100
print "Testing Accuracy: ", testing_accuracy

# Precision Outcomes
precision = precision_score(test_labels, testing_prediction, average="weighted")*100
print "Precision: ", precision

# GraphViz Output
viz_data = tree.export_graphviz(fitted_clf, out_file=None)
graph = graphviz.Source(viz_data)
# graph.render("HMEQ_Decision_Tree")

#with open("decisionTree1.txt", 'w') as f:
#    f = tree.export_graphviz(fitted_clf, out_file=f)
