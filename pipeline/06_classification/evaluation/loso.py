import numpy as np

def leave_one_record_out(records, model_train, model_predict):
    ids = list(records.keys())
    for test_id in ids:
        X_tr, y_tr = [], []
        for rid, r in records.items():
            if rid == test_id:
                continue
            X_tr.append(r["X"])
            y_tr.append(r["y"])
        if not X_tr:
            continue
        X_tr=np.vstack(X_tr)
        y_tr=np.hstack(y_tr)
        if len(np.unique(y_tr)) < 2:
            continue
        clf, scaler = model_train(X_tr, y_tr)
        y_pred = model_predict(clf, scaler, records[test_id]["X"])
        yield test_id, y_pred, records[test_id]["y"]