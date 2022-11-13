import random
import numpy as np
from numpy.random import uniform
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import kmeans_plusplus
from sklearn.mixture._gaussian_mixture import GaussianMixture, _estimate_gaussian_parameters,\
    _compute_precision_cholesky, _estimate_gaussian_covariances_full
from scipy.stats import median_abs_deviation, rankdata, weightedtau, linregress
from scipy import linalg
from itertools import combinations
from random import sample


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
    name = 'standard'

    def __init__(self, n_clusters, max_iter=100, centroid_tol=0.005, cov_tol=0.01):
        self.r = None
        self.centroids = None
        self.covariance_matrices = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = [i for i in range(self.n_clusters)]
        # pi list contains the fraction of the dataset for every cluster
        self.pi = [1 / self.n_clusters for _ in range(self.n_clusters)]
        self.labels = np.array([])
        self.centroid_tol = centroid_tol
        self.cov_tol = cov_tol

    @staticmethod
    def multivariate_normal(x, mean_vector, covariance_matrix):
        return (2 * np.pi) ** (-len(x) / 2) * np.linalg.det(covariance_matrix) ** (-1 / 2) * np.exp(
            -np.dot(np.dot((x - mean_vector).T, np.linalg.inv(covariance_matrix)), (x - mean_vector)) / 2)

    @staticmethod
    def cov(x, initial=False, **kwargs):
        if initial:
            return np.cov(x)

        return np.cov(x, **kwargs)

    @staticmethod
    def init_centroids(x):
        return [np.mean(e, axis=0) for e in x]

    def fit(self, x):
        # k-means ++
        resp = np.zeros((len(x), self.n_clusters))
        _, indices = kmeans_plusplus(
            x,
            self.n_clusters,
        )
        resp[indices, np.arange(self.n_clusters)] = 1

        _, means, __ = _estimate_gaussian_parameters(
            x, resp, 1e-6, 'full'
        )

        self.centroids = means

        # Splitting the data in n_components sub-sets
        new_x = np.array_split(x, self.n_clusters)
        self.covariance_matrices = [self.cov(e.T, initial=True) for e in new_x]
        # Deleting the new_x matrix because we will not need it anymore
        del new_x

        prev_centroids = None
        prev_cov = None
        for iteration in range(self.max_iter):
            if prev_centroids is not None and \
                    (np.linalg.norm(np.sort(self.centroids, axis=0) -
                                    np.sort(prev_centroids, axis=0)) < self.centroid_tol and
                     np.linalg.norm(np.sort(self.covariance_matrices, axis=0) -
                                    np.sort(prev_cov, axis=0)) < self.cov_tol):
                break

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

            prev_centroids = self.centroids
            prev_cov = self.covariance_matrices

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

    def probabilities(self, x):
        probas = []
        for n in range(len(x)):
            probas.append([self.multivariate_normal(x[n], self.centroids[k], self.covariance_matrices[k])
                           for k in range(self.n_clusters)])

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


# class SpearmanGaussianMixtures(GaussianMixtures):
#     name = 'spearman'
#
#     @staticmethod
#     def _sigma(x):
#         return np.std(x, axis=1).reshape(1, len(x))
#
#     def _cov(self, x, w=None):
#
#         ranked_x = rankdata(x, axis=1)
#         if w is None:
#             w = np.ones(len(ranked_x[0]))
#
#         c = np.cov(ranked_x, aweights=w, ddof=0)
#         diag = np.diag(c).reshape(len(x), 1)
#         correlation = c / np.sqrt(diag @ diag.T)
#
#         sigma = self._sigma(x)
#         sigma_matrix = sigma.T @ sigma
#
#         return sigma_matrix * correlation
#
#     def cov(self, x, initial=False, aweights=None, **kwargs):
#         if initial:
#             return self._cov(x)
#
#         return self._cov(x, aweights)


class StandardGaussianMixture(GaussianMixture):
    name = 'standard'

    def _estimate_gaussian_covariances_full(self, resp, x, nk, means, reg_covar):
        return _estimate_gaussian_covariances_full(resp, x, nk, means, reg_covar)

    def _estimate_gaussian_parameters(self, x, resp, reg_covar):
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, x) / nk[:, np.newaxis]
        covariances = self._estimate_gaussian_covariances_full(resp, x, nk, means, reg_covar)
        return nk, means, covariances

    def _initialize(self, x, resp):
        n_samples, _ = x.shape

        weights, means, covariances = self._estimate_gaussian_parameters(
            x, resp, self.reg_covar
        )
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init
        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        elif self.covariance_type == "full":
            self.precisions_cholesky_ = np.array(
                [
                    linalg.cholesky(prec_init, lower=True)
                    for prec_init in self.precisions_init
                ]
            )

    def _m_step(self, x, log_resp):
        self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(
            x, np.exp(log_resp), self.reg_covar
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def accuracy_score(self, x, true_labels):
        c_matrix = contingency_matrix(true_labels, self.predict(x))
        return c_matrix.max(axis=0).sum() / c_matrix.sum()


class SpearmanGaussianMixture(StandardGaussianMixture):
    name = 'spearman'

    @staticmethod
    def _sigma(x):
        return np.std(x, axis=0).reshape(1, x.shape[1])

    def _estimate_gaussian_covariances_full(self, resp, x, nk, means, reg_covar):
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features))
        ranked_x = rankdata(x, axis=0)
        for k in range(n_components):
            c = np.cov(ranked_x.T, aweights=resp[:, k], ddof=0)
            diag = np.diag(c).reshape(n_features, 1)
            correlation = c / np.sqrt(diag @ diag.T)

            sigma = self._sigma(x)
            sigma_matrix = sigma.T @ sigma
            covariances[k] = sigma_matrix * correlation / nk[k]
            covariances[k].flat[:: n_features + 1] += reg_covar
        return covariances


# class MADSpearmanGaussianMixtures(SpearmanGaussianMixtures):
#     name = 'mad_spearman'
#
#     @staticmethod
#     def _sigma(x):
#         return median_abs_deviation(x, axis=1).reshape(1, len(x))


class MADSpearmanGaussianMixture(SpearmanGaussianMixture):
    name = 'mad_spearman'

    @staticmethod
    def _sigma(x):
        return median_abs_deviation(x, axis=0).reshape(1, x.shape[1])


# class KendallGaussianMixtures(SpearmanGaussianMixtures):
#     name = 'kendall'
#
#     @staticmethod
#     def _corrcoef(x, w, i, j):
#         return weightedtau(x=x[i, :], y=x[j, :], rank=range(x.shape[1]),
#                            weigher=lambda idx: w[idx], additive=False)[0]
#
#     def _cov(self, x, w=None):
#         n_features = len(x)
#         if w is None:
#             w = [1] * x.shape[1]
#
#         correlation = np.ones((n_features, n_features))
#         for i, j in combinations(range(n_features), 2):
#             coefficient = self._corrcoef(x, w, i, j)
#             correlation[i, j] = coefficient
#             correlation[j, i] = coefficient
#
#         sigma = self._sigma(x)
#         sigma_matrix = sigma.T @ sigma
#
#         return sigma_matrix * correlation


class KendallGaussianMixture(SpearmanGaussianMixture):
    name = 'kendall'

    @staticmethod
    def _corrcoef(x, w, i, j):
        return weightedtau(x=x[i, :], y=x[j, :], rank=range(x.shape[1]),
                           weigher=lambda idx: w[idx], additive=False)[0]

    def _cov(self, resp, x, nk, means, reg_covar):
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            correlation = np.ones((n_features, n_features))
            for i, j in combinations(range(n_features), 2):
                coefficient = self._corrcoef(x, resp[:, k], i, j)
                correlation[i, j] = coefficient
                correlation[j, i] = coefficient

            sigma = self._sigma(x)
            sigma_matrix = sigma.T @ sigma
            covariances[k] = sigma_matrix * correlation / nk[k]
            covariances[k].flat[:: n_features + 1] += reg_covar
        return covariances


class MADKendallGaussianMixture(KendallGaussianMixture, MADSpearmanGaussianMixture):
    name = 'mad_kendall'


class OrtizGaussianMixture(KendallGaussianMixture):
    name = 'ortiz'
    initial_limit = 10

    @staticmethod
    def get_combinations(combis, w_combis):
        return zip(combis, w_combis)

    def _corrcoef(self, x, w, i, j):
        combis = list(combinations(x.T, 2))
        w_combis = list(combinations(w, 2))
        limit = self.initial_limit
        for i in range(5):
            s1 = 0
            s2 = 0
            for combi, (w1, w2) in self.get_combinations(combis, w_combis):
                pair = np.vstack(combi)
                slope = linregress(pair[:, i], pair[:, j]).slope
                if not -limit < slope < limit:
                    continue
                s1 += slope * w1 * w2
                s2 += abs(slope) * w1 * w2

            if s2 != 0:
                break

            limit *= 10

        # noinspection PyUnboundLocalVariable
        return s1 / s2


class MADOrtizGaussianMixture(OrtizGaussianMixture, MADSpearmanGaussianMixture):
    name = 'mad_ortiz'


class ApproxOrtizGaussianMixture(OrtizGaussianMixture):
    name = 'approx'

    @staticmethod
    def get_combinations(combis, w_combis):
        return sample(list(zip(combis, w_combis)), 1000 if len(combis) > 1000 else len(combis))


class MADApproxOrtizGaussianMixture(ApproxOrtizGaussianMixture, MADSpearmanGaussianMixture):
    name = 'mad_approx'

