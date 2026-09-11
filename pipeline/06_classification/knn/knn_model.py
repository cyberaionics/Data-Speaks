from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def train(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(Xs, y_train)
    return clf, scaler

def predict(clf, scaler, X):
    return clf.predict(scaler.transform(X))