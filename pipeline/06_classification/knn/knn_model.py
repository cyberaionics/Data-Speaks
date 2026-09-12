from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

K_NEIGHBORS = 5

def train(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, n_jobs=-1)
    clf.fit(Xs, y_train)
    return clf, scaler

def predict(clf, scaler, X):
    return clf.predict(scaler.transform(X))