import random
import numpy as np
from numpy.random import uniform
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture._gaussian_mixture import GaussianMixture,\
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


class MADSpearmanGaussianMixture(SpearmanGaussianMixture):
    name = 'mad_spearman'

    @staticmethod
    def _sigma(x):
        return median_abs_deviation(x, axis=0).reshape(1, x.shape[1])


class KendallGaussianMixture(SpearmanGaussianMixture):
    name = 'kendall'

    @staticmethod
    def _corrcoef(x, w, i, j):
        return weightedtau(x=x[i, :], y=x[j, :], rank=range(x.shape[1]),
                           weigher=lambda idx: w[idx], additive=False)[0]

    def _estimate_gaussian_covariances_full(self, resp, x, nk, means, reg_covar):
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
        combis = list(combinations(x, 2))
        w_combis = list(combinations(w, 2))
        limit = self.initial_limit
        for _ in range(5):
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

