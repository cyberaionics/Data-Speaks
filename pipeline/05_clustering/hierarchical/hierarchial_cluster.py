from sklearn.cluster import AgglomerativeClustering

def hierarchical_labels(X, n_clusters=2):
    return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X)