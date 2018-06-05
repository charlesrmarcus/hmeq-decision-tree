import pandas as pd
import numpy as np
from scipy.stats import kurtosis
import graphviz
import matplotlib.pyplot as plt
import itertools
from sklearn import linear_model, decomposition
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.random_projection import SparseRandomProjection

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

train, test, train_labels, test_labels = train_test_split(hmeq_features, hmeq_target, test_size=0.33, random_state=42)

# Finding the Optimal Component
complist = [2, 3, 4, 5, 6, 7, 8, 9, 10]

### PCA ###
for n_comp in complist:
    # Fit the PCA Analysis
    result = PCA(n_components=n_comp).fit(train)
    #print("Component Number: " + str(n_comp))
    #print("Components: " + str(result.components_))
    #print("Explained Variance: " + str(result.explained_variance_))
    #print("Explained Variance Ratio: " + str(result.explained_variance_ratio_))
    #print("PCA Score: " + str(result.score(train)))

# PCA Visualization
logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

pca.fit(hmeq_features)

plt.figure(1, figsize=(5, 4))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# Prediction
n_components = np.int32(np.linspace(2, 10, 2))
Cs = np.logspace(-4, 5, 4)

estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(hmeq_features, hmeq_target)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.suptitle("Explained Variance vs. # Comp for HMEQ")
#plt.show()

### Random Projections ###
accuracies = []
components = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Create Linear SVC Model
model = LinearSVC()
model.fit(train, train_labels)
baseline = accuracy_score(model.predict(test), test_labels)

# loop over the projection sizes
for comp in components:
    # create the random projection
    sp = SparseRandomProjection(n_components=comp)
    X = sp.fit_transform(train)

    # train a classifier on the sparse random projection
    model = LinearSVC()
    model.fit(X, train_labels)

    # evaluate the model and update the list of accuracies
    transformed_test = sp.transform(test)
    accuracies.append(accuracy_score(model.predict(transformed_test), test_labels))

# create the figure
plt.figure()
plt.suptitle("Accuracy of Sparse Projection on HMEQ")
plt.xlabel("# of Components")
plt.ylabel("Accuracy")
# Change to Match Num of Features in Dataset
plt.xlim([2, 10])
plt.ylim([0, 1.0])

# plot the baseline and random projection accuracies
plt.plot(components, [baseline] * len(accuracies), color="r")
plt.plot(components, accuracies)
#plt.show()

#### Fast ICA ####
ica = decomposition.FastICA(n_components=10, algorithm='parallel', tol=1e+1, max_iter=1000000)
S_ = ica.fit_transform(hmeq_features)
#print("Kurtosis: " + str(kurtosis(S_)))

#### NMF ####
pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', LinearSVC())
])

N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7), FastICA(), SparseRandomProjection(), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]

reducer_labels = ['PCA', 'ICA', 'Random Projection', 'NMF']

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
grid.fit(hmeq_features, hmeq_target)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques - HMEQ")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('HMEQ classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')
plt.show()

# Accuracy Model for NMF
accuracies = []
# Change to Match Num of Features in Dataset
components = np.int32(np.linspace(2, 10, 2))

# Create Linear SVC Model
model = LinearSVC()
model.fit(train, train_labels)
baseline = accuracy_score(model.predict(test), test_labels)

# loop over the projection sizes
for comp in components:
    # create the random projection
    nmf = NMF(n_components=comp)
    X = nmf.fit_transform(train)

    # train a classifier on the sparse random projection
    model = LinearSVC()
    model.fit(X, train_labels)

    # evaluate the model and update the list of accuracies
    transformed_test = nmf.transform(test)
    accuracies.append(accuracy_score(model.predict(transformed_test), test_labels))

# create the figure
plt.figure()
plt.suptitle("Accuracy of NMF on BRC")
plt.xlabel("# of Components")
plt.ylabel("Accuracy")
# Change to Match Num of Features in Dataset
plt.xlim([2, 10])
plt.ylim([0, 1.0])

# plot the baseline and random projection accuracies
plt.plot(components, [baseline] * len(accuracies), color="r")
plt.plot(components, accuracies)
plt.show()
