import numpy as np
from numpy.random import uniform
from sklearn.metrics.cluster import contingency_matrix
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

        self.labels = self.predict(x_train)

    def predict(self, x):
        centroids = []
        centroid_ids = []
        self.inertia = 0
        for point in x:
            dists = euclidean(point, self.centroids)
            centroid_id = np.argmin(dists)
            self.inertia += dists[centroid_id] ** 2
            centroids.append(self.centroids[centroid_id])
            centroid_ids.append(centroid_id)
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
        x_space, x = empiric(x, **kwargs)
        step = x_space[1] - x_space[0]
        positive_area = sum((1 - x[x_space >= 0]) * step)
        negative_area = sum(x[x_space < 0] * step)
        return positive_area - negative_area

    return 0


class RobustKMeans(KMeans):
    @staticmethod
    def find_means(sorted_points, **kwargs):
        return np.array([np.apply_along_axis(choquet_mean, 0, cluster, **kwargs) for cluster in sorted_points])
