from sklearn.metrics import silhouette_score, adjusted_rand_score

def silhouette(X, labels):
    if len(set(labels)) < 2:
        return None
    return float(silhouette_score(X, labels))

def ari(y_true, y_pred):
    return float(adjusted_rand_score(y_true, y_pred))