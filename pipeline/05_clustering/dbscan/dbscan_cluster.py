from sklearn.cluster import DBSCAN

def dbscan_labels(X, eps = 1.5, min_samples=10):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)