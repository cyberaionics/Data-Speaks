from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

SVM_PARAMS = dict(kernel="rbf", gamma=0.1, C=1.0, class_weight="balanced", cache_size=500)

def train(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = SVC(**SVM_PARAMS)
    clf.fit(Xs, y_train)
    return clf, scaler

def predict(clf, scaler, X):
    return clf.predict(scaler.transform(X))