from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    clf.fit(Xs, y_train)
    return clf, scaler

def predict(clf, scaler, X):
    return clf.predict(scaler.transform(X))