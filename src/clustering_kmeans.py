import pandas as pd
import numpy as np
from scipy.stats import kurtosis
import graphviz
import matplotlib.pyplot as plt
import itertools
from sklearn import linear_model, decomposition
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.random_projection import SparseRandomProjection
import scikitplot as skplt

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

# Split out data
train, test, train_labels, test_labels = train_test_split(hmeq_features, hmeq_target, test_size=0.33, random_state=42)

# Initial Model Classifier
clf = KMeans(n_clusters=2, init='k-means++', n_init=5, max_iter=500)
fitted_clf = clf.fit(train, train_labels)

# Cross Validation
scores = cross_val_score(fitted_clf, hmeq_features, hmeq_target, cv=5)
print scores

# Training and Testing
training_prediction = clf.predict(train)
training_accuracy = accuracy_score(train_labels, training_prediction)*100
print "Training Accuracy is", training_accuracy
test_prediction = clf.predict(test)
print "Testing Accuracy is", accuracy_score(test_labels, test_prediction)*100

skplt.estimators.plot_learning_curve(fitted_clf, hmeq_features, hmeq_target, title="Learning Curve: k-Means")
plt.show()

# A demo of K-Means clustering on the handwritten digits data
# http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
# Change Dataset to alter cluster diagrams!
np.random.seed(42)
data = scale(brc.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(brc.target))
labels = brc.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))
print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             sklearn.metrics.homogeneity_score(labels, estimator.labels_),
             sklearn.metrics.completeness_score(labels, estimator.labels_),
             sklearn.metrics.v_measure_score(labels, estimator.labels_),
             sklearn.metrics.adjusted_rand_score(labels, estimator.labels_),
             sklearn.metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             sklearn.metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
#ica = FastICA(n_components=n_digits).fit(data)
#srp = SparseRandomProjection(n_components=n_digits).fit(data)
# For Sparse Random Projections, use srp.components_.toarray()
#nmf = NMF(n_components=n_digits).fit(np.absolute(data))
# For NMF, be sure to use np.absolute to get ride of negative values

bench_k_means(KMeans(init=np.absolute(pca.components_), n_clusters=n_digits, n_init=1),
              name="NMF-based",
              data=data)
print(82 * '_')

# Visualize the results on PCA-reduced data

reduced_data = NMF(n_components=2).fit_transform(np.absolute(data))
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
#kmeans.fit(reduced_data)
kmeans.fit(data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
#x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
#y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

#plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the HMEQ dataset \n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
