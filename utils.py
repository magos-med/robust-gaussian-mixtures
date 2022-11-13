import numpy as np
import plotly.express as px

from clustering import (
    StandardGaussianMixture,
    SpearmanGaussianMixture,
    MADSpearmanGaussianMixture,
    KendallGaussianMixture,
    MADKendallGaussianMixture,
    OrtizGaussianMixture,
    MADOrtizGaussianMixture,
    ApproxOrtizGaussianMixture,
    MADApproxOrtizGaussianMixture,
)

ALL_MODELS = (
    StandardGaussianMixture,
    SpearmanGaussianMixture,
    MADSpearmanGaussianMixture,
    KendallGaussianMixture,
    MADKendallGaussianMixture,
    OrtizGaussianMixture,
    MADOrtizGaussianMixture,
    ApproxOrtizGaussianMixture,
    MADApproxOrtizGaussianMixture,
)


ellipse_color = 'rgb(55.0, 111.5, 155.0)'


def ellipse(x_center=0, y_center=0, ax1=(0, 1), ax2=(1, 0), a=1, b=1, n=100):
    if np.linalg.norm(ax1).round(6) != 1 or np.linalg.norm(ax2).round(6) != 1:
        raise ValueError('ax1, ax2 must be unit vectors')
    if abs(np.dot(ax1, ax2)) > 1e-06:
        raise ValueError('ax1, ax2 must be orthogonal vectors')
    t = np.linspace(0, 2 * np.pi, n)
    xs = a * np.cos(t)
    ys = b * np.sin(t)
    r = np.array([ax1, ax2]).T
    xp, yp = np.dot(r, [xs, ys])
    x = xp + x_center
    y = yp + y_center
    return x, y


def plot_gaussian_mixtures(gmm, x_train_scaled):
    fig = px.scatter(x=gmm.means_[:, 0], y=gmm.means_[:, 1])
    fig.update_traces(mode='markers', marker_size=12, marker_color='black', marker_symbol='x-thin', marker_line_width=2)
    is_first = True
    for centroid, cov in zip(gmm.means_, gmm.covariances_):
        v, w = np.linalg.eigh(cov)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])

        for scale, opacity in zip((1, 2, 4), (0.4, 0.2, 0.05)):
            if is_first:
                legend = True
                is_first = False
            else:
                legend = False
            x, y = ellipse(x_center=centroid[0], y_center=centroid[1],
                           ax1=[np.cos(angle), np.sin(angle)], ax2=[-np.sin(angle), np.cos(angle)],
                           a=v[0] * scale, b=v[1] * scale)
            fig.add_scatter(
                x=x, y=y, mode='none', fill='toself', opacity=opacity, fillcolor=ellipse_color,
                legendgroup='ellipses', name='ellipses', showlegend=legend
            )

    fig.add_scatter(x=x_train_scaled[:, 0], y=x_train_scaled[:, 1], marker_color=gmm.predict(x_train_scaled),
                    mode='markers', name='points')
    return fig
