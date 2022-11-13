import numpy as np
import pandas as pd
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from itertools import combinations
from traceback import print_exc
from multiprocessing import Pool, Manager
from itertools import repeat

from utils import ALL_MODELS


OUTPUT_FILE = 'output/test_results.csv'


def calculate_distance(x_train_scaled, true_labels, n_clusters):
    centroids = []
    for c in range(n_clusters):
        centroids.append(x_train_scaled[true_labels == c].mean(axis=0))

    avg_distance = 0
    combis = list(combinations(centroids, 2))
    for c1, c2 in combis:
        avg_distance += np.linalg.norm(c1 - c2) / len(combis)

    return avg_distance


def test_method(model, results, x_train_scaled, true_labels, x_train_noise_scaled, true_labels_noise, n_clusters):
    try:
        gmm = model(n_components=n_clusters, max_iter=100)
        gmm.fit(x_train_scaled)
        score = gmm.accuracy_score(x_train_scaled, true_labels)
        results[f'{model.name}_score'] = score

        gmm_noise = model(n_components=n_clusters, max_iter=100)
        gmm_noise.fit(x_train_noise_scaled)
        score_noise = gmm_noise.accuracy_score(x_train_noise_scaled, true_labels_noise)
        results[f'{model.name}_score_noise'] = score_noise

        results[f'{model.name}_centroid_variation'] = np.linalg.norm(gmm.means_ - gmm_noise.means_)

        total = 0
        for cov, cov_noise in zip(gmm.covariances_, gmm_noise.covariances_):
            variation = np.linalg.norm(cov / np.linalg.norm(cov) - cov_noise / np.linalg.norm(cov_noise))
            total += variation
        results[f'{model.name}_cov_variation'] = total
        print(results)

    except Exception as ex:
        print(ex)
        print_exc()


def test_all_methods(x_train_scaled, true_labels, x_train_noise_scaled, true_labels_noise,
                     n_clusters, df):
    t0 = time.perf_counter()
    mgr = Manager()
    results = mgr.dict()
    pool = Pool(processes=9)
    pool.starmap(
        test_method,
        zip(
            ALL_MODELS,
            repeat(results),
            repeat(x_train_scaled),
            repeat(true_labels),
            repeat(x_train_noise_scaled),
            repeat(true_labels_noise),
            repeat(n_clusters),
        )
    )

    pool.close()
    pool.join()

    for col in results:
        df[col] = results[col]
    t1 = time.perf_counter()
    df['time'] = round(t1 - t0)

    if not os.path.isfile(OUTPUT_FILE):
        df.to_csv(OUTPUT_FILE, index=False)
    else:
        df_old = pd.read_csv(OUTPUT_FILE)
        df = pd.concat([df_old, df])
        df.to_csv(OUTPUT_FILE, index=False)


def main(n_tests):
    _n_samples = np.random.choice([100, 1000, 10000, 100000], p=[0.7, 0.1, 0.1, 0.1], size=n_tests)
    _n_clusters = np.random.choice(range(3, 20), size=n_tests)
    boxes_size = np.random.uniform(1.5, 20, n_tests)
    _noise_proportion = np.random.choice([0.01, 0.05, 0.1], size=n_tests)
    _std = np.random.choice([0.5, 1, 2, 3], size=n_tests)
    _noise_type = np.random.choice(['specific', 'random'], size=n_tests)
    _cluster_type = np.random.choice(
        ['standard', 'anisotropic', 'variance_diff', 'unbalanced'],
        p=[0.7, 0.1, 0.1, 0.1],
        size=n_tests
    )
    _n_features = np.random.choice(
        list(range(2, 20)) + [100],
        size=n_tests
    )

    variations = [
        _n_samples,
        _n_clusters,
        boxes_size,
        _std,
        _noise_type,
        _cluster_type,
        _n_features
    ]

    for n_samples, n_clusters, box_size, std, noise_type, cluster_type, n_features in zip(*variations):
        try:
            if cluster_type == 'variance_diff':
                std = np.random.uniform(0.5, 2.5, n_clusters)

            x_train, true_labels = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=n_clusters,
                cluster_std=std,
                center_box=[-box_size, box_size],
            )

            if cluster_type == 'anisotropic' and n_features == 2:
                theta = np.radians(np.random.randint(20, 50))
                t = np.tan(theta)
                if np.random.choice([True, False]):
                    shear = np.array(((1, t), (0, 1))).T
                else:
                    shear = np.array(((1, 0), (t, 1))).T
                x_train = x_train.dot(shear)

            elif cluster_type == 'unbalanced':
                x_train = np.vstack(list(
                    x_train[true_labels == i][:int(n_samples / n_clusters / (i + 1)) + 2]
                    for i in range(n_clusters)
                ))
                true_labels = np.vstack(list(
                    true_labels[true_labels == i][:int(n_samples / n_clusters / (i + 1)) + 2].reshape(-1, 1)
                    for i in range(n_clusters)
                )).flatten()

            x_train_scaled = StandardScaler().fit_transform(x_train)

            avg_distance = round(calculate_distance(x_train_scaled, true_labels, n_clusters), 3)

            if noise_type == 'random':
                n_noisy_samples = int(n_samples * np.random.choice([0.01, 0.05, 0.1]))
                noise = (np.random.random((n_noisy_samples, n_features)) - 0.5) * box_size * 2
                noise_labels = np.random.randint(0, n_clusters, n_noisy_samples)

            else:
                noise, noise_labels = make_blobs(
                    n_samples=np.random.randint(1, 5),
                    n_features=n_features,
                    centers=[np.random.uniform(box_size * 1.2, box_size * 2, n_features)]
                )

            x_train_noise = np.append(x_train, noise, axis=0)
            true_labels_noise = np.append(true_labels, noise_labels)
            x_train_noise_scaled = StandardScaler().fit_transform(x_train_noise)

            df = pd.DataFrame({
                'n_features': [n_features],
                'n_clusters': [n_clusters],
                'n_samples': [n_samples],
                'box_size': [box_size],
                'std': [std if cluster_type != 'variance_diff' else np.nan],
                'noise_type': [noise_type],
                'cluster_type': [cluster_type],
                'avg_distance': [avg_distance],
            })

            test_all_methods(
                x_train_scaled,
                true_labels,
                x_train_noise_scaled,
                true_labels_noise,
                n_clusters,
                df
            )

        except Exception as ex:
            print(ex)
            print_exc()


if __name__ == '__main__':
    main(n_tests=10000)
