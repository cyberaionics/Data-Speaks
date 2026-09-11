from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(Xs, y_train)
    return clf, scaler

def predict(clf, scaler, X):
    return clf.predict(scaler.transform(X))