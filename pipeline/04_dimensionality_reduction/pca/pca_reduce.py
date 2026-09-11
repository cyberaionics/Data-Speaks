from sklearn.decomposition import PCA

def pca_project(X, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X)
    return Z, pca, pca.explained_variance_ratio_