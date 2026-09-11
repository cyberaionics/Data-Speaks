from sklearn.cluster import KMeans

def kmeans_label(X, n_clusters = 2, random_state = 42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(X), km.cluster_centers_