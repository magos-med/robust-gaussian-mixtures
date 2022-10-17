import numpy as np
from numpy.random import uniform
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import spearmanr, median_abs_deviation
import random


def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = np.array([])
        self.inertia = None

    @staticmethod
    def find_means(sorted_points, **_):
        return np.array([np.mean(cluster, axis=0) for cluster in sorted_points])

    def fit(self, x_train, **kwargs):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(x_train)]
        for _ in range(self.n_clusters - 1):
            # Calculate distances from points to the centroids
            dists = np.sum([euclidean(centroid, x_train) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(x_train)), size=1, p=dists)
            self.centroids += [x_train[new_centroid_idx]]

        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in x_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = self.find_means(sorted_points, **kwargs)
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

        self.labels = self.predict(x_train, **kwargs)

    def predict(self, x, **kwargs):
        centroids = []
        centroid_ids = []
        sorted_points = [[] for _ in range(self.n_clusters)]
        self.inertia = 0
        for point in x:
            dists = euclidean(point, self.centroids)
            centroid_id = np.argmin(dists)
            sorted_points[centroid_id].append(point)
            self.inertia += dists[centroid_id] ** 2
            centroids.append(self.centroids[centroid_id])
            centroid_ids.append(centroid_id)

        self.centroids = self.find_means(sorted_points, **kwargs)
        return np.array(centroid_ids)

    def score(self, labels):
        c_matrix = contingency_matrix(labels, self.labels)
        return c_matrix.max(axis=0).sum() / c_matrix.sum()


def empiric(x, n=100):
    x_space = np.linspace(x.min(), x.max(), n)
    probabilities = []
    for t in x_space:
        n = len(x)
        i = len(x[x < t])
        p = i / n
        probabilities.append(p)
    return x_space, np.array(probabilities)


def choquet_mean(x, **kwargs):
    if len(x) != 0:  # Prevent empty cluster error
        x_min = x.min()
        positive_square = x_min if x_min > 0 else 0
        x_max = x.max()
        negative_square = x_max if x_max < 0 else 0
        x_space, x = empiric(x, **kwargs)
        step = x_space[1] - x_space[0]
        positive_area = sum((1 - x[x_space >= 0]) * step) + positive_square
        negative_area = sum(x[x_space < 0] * step) - negative_square
        return positive_area - negative_area

    return np.array([np.nan, np.nan])


class RobustKMeans(KMeans):
    @staticmethod
    def find_means(sorted_points, **kwargs):
        return np.array([np.apply_along_axis(choquet_mean, 0, cluster, **kwargs) for cluster in sorted_points])


class GaussianMixtures:
    def __init__(self, n_clusters, max_iter=100):
        self.r = None
        self.centroids = None
        self.covariance_matrices = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = [i for i in range(self.n_clusters)]
        # pi list contains the fraction of the dataset for every cluster
        self.pi = [1 / self.n_clusters for _ in range(self.n_clusters)]
        self.labels = np.array([])

    @staticmethod
    def multivariate_normal(x, mean_vector, covariance_matrix):
        return (2 * np.pi) ** (-len(x) / 2) * np.linalg.det(covariance_matrix) ** (-1 / 2) * np.exp(
            -np.dot(np.dot((x - mean_vector).T, np.linalg.inv(covariance_matrix)), (x - mean_vector)) / 2)

    @staticmethod
    def cov(x, initial=False, **kwargs):
        if initial:
            return np.cov(x)

        return np.cov(x, **kwargs)

    def fit(self, x):
        # Splitting the data in n_components sub-sets
        new_x = np.array_split(x, self.n_clusters)
        # Initial computation of the mean-vector and covariance matrix
        self.centroids = [np.mean(e, axis=0) for e in new_x]  # ToDo: Initialize with empiric mean
        self.covariance_matrices = [self.cov(e.T, initial=True) for e in new_x]
        # Deleting the new_x matrix because we will not need it anymore
        del new_x

        for iteration in range(self.max_iter):
            ''' ----------------   E - STEP   ------------------ '''
            # Initiating the r matrix, every row contains the probabilities
            # for every cluster for this row
            self.r = np.zeros((len(x), self.n_clusters))
            # Calculating the r matrix
            for i in range(len(x)):
                for k in range(self.n_clusters):
                    self.r[i][k] = self.pi[k] * self.multivariate_normal(x[i], self.centroids[k],
                                                                         self.covariance_matrices[k])
                    self.r[i][k] /= sum(
                        [self.pi[j] * self.multivariate_normal(x[i], self.centroids[j], self.covariance_matrices[j])
                         for j in range(self.n_clusters)])
            # Calculating the N
            n = np.sum(self.r, axis=0)

            ''' ---------------   M - STEP   --------------- '''
            # Initializing the mean vector as a zero vector
            self.centroids = np.zeros((self.n_clusters, len(x[0])))
            # Updating the mean vector
            for k in range(self.n_clusters):
                for i in range(len(x)):
                    self.centroids[k] += self.r[i][k] * x[i]

            self.centroids = [1 / n[k] * self.centroids[k] for k in range(self.n_clusters)]
            # Initiating the list of the covariance matrices
            self.covariance_matrices = [np.zeros((len(x[0]), len(x[0]))) for _ in range(self.n_clusters)]
            # Updating the covariance matrices
            for k in range(self.n_clusters):
                self.covariance_matrices[k] = self.cov(x.T, aweights=self.r[:, k], ddof=0)
            self.covariance_matrices = [1 / n[k] * self.covariance_matrices[k] for k in range(self.n_clusters)]
            # Updating the pi list
            self.pi = [n[k] / len(x) for k in range(self.n_clusters)]

        self.labels = self.predict(x)
        self.centroids = np.array(self.centroids)

    def predict(self, x):
        probas = []
        for n in range(len(x)):
            probas.append([self.multivariate_normal(x[n], self.centroids[k], self.covariance_matrices[k])
                           for k in range(self.n_clusters)])
        cluster = []
        for proba in probas:
            cluster.append(self.clusters[proba.index(max(proba))])
        return cluster

    def score(self, labels):
        c_matrix = contingency_matrix(labels, self.labels)
        return c_matrix.max(axis=0).sum() / c_matrix.sum()


class SpearmanGaussianMixtures(GaussianMixtures):
    @staticmethod
    def _sigma(x):
        return np.std(x, axis=1).reshape(1, len(x))

    def _cov(self, x):
        corr = spearmanr(x.T)
        if type(corr[0]) == np.float64:
            correlation = np.array(
                [[1, corr[0]],
                 [corr[0], 1]]
            )
        else:
            correlation = corr[0]

        sigma = self._sigma(x)
        sigma_matrix = sigma.T @ sigma

        return sigma_matrix * correlation

    def cov(self, x, initial=False, aweights=None, **kwargs):
        if initial:
            return self._cov(x)

        covariance = self._cov(x)
        return np.cov(x, **kwargs)  # ToDo: Change this one properly


class MADSpearmanGaussianMixtures(SpearmanGaussianMixtures):
    @staticmethod
    def _sigma(x):
        return median_abs_deviation(x, axis=1).reshape(1, len(x))
