import numpy as np

REFRACTORY_SEC = 30.0

def detect_events(y_pred, starts, refractory = REFRACTORY_SEC):
    events = []
    last = -np.inf
    prev = 0
    for pred, t in zip(y_pred, starts):
        if pred == 1 and prev == 0 and (t-last) > refractory:
            events.append(float(t))
            last = t
        prev = pred
    return events

def sensitivity(seizures, events, tol=5.0):
    if not seizures:
        return None
    for s0, s1 in seizures:
        for t in events:
            if s0-tol <= t <= s1 + tol:
                return 1.0
    return 0.0

def latency(seizures, events, tol = 5.0):
    if not seizures:
        return None
    onset = min(s0 for s0, _ in seizures)
    for t in events:
        if t >= onset - tol:
            return max(0.0, t-onset)
    return None

def false_detection_per_24h(events, record_duration_sec):
    if record_duration_sec <= 0:
        return 0.0
    return len(events) * 24 * 3600 / record_duration_sec