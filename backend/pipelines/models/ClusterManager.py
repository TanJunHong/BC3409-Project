from sklearn.cluster import KMeans, OPTICS, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest

import numpy as np
import pandas as pd


class ClusterManager:
    """
    Provides a wrapper to initialize, fit and predict a clustering model based on the input data. It then caches the
    model for any recording purposes.

    In this class, the functions that wrap the cluster methods would set our defaults to be used in the pipeline.
    Each method has a reference to resources used to get optimal defaults.
    """

    def __init__(self):
        self.kmeans = None
        self.dbscan = None
        self.optics = None
        self.hierarchical = None
        self.anomaly = None

        self.predict_algorithms = {
            "kmeans": self.predict_kmeans,
            "soft_kmeans": self.predict_kmeans,
            "dbscan": self.predict_dbscan,
            "optics": self.predict_optics,
            "hierarchical": self.predict_hierarchical,
            "anomaly": self.anomaly_detection
        }

    def predict_kmeans(self, reduced_vec: np.ndarray, num_clusters: int = 7) -> (pd.Series, KMeans):
        self.kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)
        self.kmeans.fit_transform(reduced_vec)
        return self.kmeans.labels_, self.kmeans

    def predict_soft_kmeans(self, reduced_vec: np.ndarray, num_clusters: int, threshold: float = 0.5) -> (list, KMeans):
        # Work on implementing https://www.cs.cmu.edu/~02251/recitations/recitation_soft_clustering.pdf to convert to prob
        _, kmeans = self.predict_kmeans(reduced_vec, num_clusters)
        dist = kmeans.transform(processed)
        label_sorted = dist.argsort(axis=1)
        dist.sort()

        soft_labels_dist = []
        for idx, row in enumerate(dist):
            row_labels = label_sorted[idx, :]
            row_labels = [(dist, label) for dist, label in zip(row, row_labels) if dist < threshold]
            row_labels = row_labels[::]

            soft_labels_dist.append(np.array(row_labels))
        return soft_labels_dist, self.kmeans

    def predict_dbscan(self, reduced_vec: np.ndarray, eps: float = 0.5, min_samples: int = None) -> (pd.Series, DBSCAN):

        # Selection of eps and min_samples
        # https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd

        dim = reduced_vec.shape[1]

        self.dbscan = DBSCAN(eps, min_samples=min_samples) if min_samples else DBSCAN(eps)
        self.dbscan.fit_predict(reduced_vec)
        return self.dbscan.labels_, self.dbscan

    def predict_optics(self, reduced_vec: np.ndarray, min_samples: int) -> (np.ndarray, OPTICS):
        # https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd
        # https://github.com/christianversloot/machine-learning-articles/blob/main/performing-optics-clustering-with-python-and-scikit-learn.md
        self.optics = OPTICS(min_samples=min_samples)
        self.optics.fit_predict(reduced_vec)
        return self.optics.labels_, self.optics

    def predict_hierarchical(self, reduced_vec: np.ndarray, num_clusters: int = 10) -> (np.ndarray, AgglomerativeClustering):
        # https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad#:~:text=To%20get%20the%20optimal%20number,the%20distance%20between%20those%20clusters.
        # Metrics to determine num_clusters for AgglomerativeClustering
        self.hierarchical = AgglomerativeClustering(num_clusters)
        self.hierarchical.fit_predict(reduced_vec)
        return self.hierarchical.labels_, self.hierarchical

    def anomaly_detection(self, reduced_vec: np.ndarray, num_trees: int = 250) -> (np.ndarray, IsolationForest):
        self.anomaly = IsolationForest(n_estimators=num_trees)
        anomaly_score = self.anomaly.fit_predict(reduced_vec)
        return anomaly_score, self.anomaly


if __name__ == "__main__":
    processed = np.load("../../temp_data/brown_lsa_data.npy", allow_pickle=False)

    clusterManager = ClusterManager()

    labels, model = clusterManager.predict_kmeans(processed, 10)
    print(labels)
    np.save("../../temp_data/kmeans_data.npy", labels, allow_pickle=False)

    labels, model = clusterManager.predict_soft_kmeans(processed, 10)
    print(labels)
    np.save("../../temp_data/soft_kmeans_data.npy", labels, allow_pickle=True)

    labels, model = clusterManager.predict_dbscan(processed, 3)
    print(labels)
    np.save("../../temp_data/dbscan_data.npy", labels, allow_pickle=False)

    labels, model = clusterManager.predict_optics(processed, 10)
    print(labels)
    np.save("../../temp_data/optics_data.npy", labels, allow_pickle=False)

    labels, model = clusterManager.predict_hierarchical(processed, 10)
    print(labels)
    np.save("../../temp_data/hierarchical_data.npy", labels, allow_pickle=False)

    labels, model = clusterManager.anomaly_detection(processed)
    print(labels)
    np.save("../../temp_data/anomaly_data.npy", labels, allow_pickle=False)
