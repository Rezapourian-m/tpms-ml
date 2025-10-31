import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def regression_scores(y_true, y_pred, names=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n_targets = y_true.shape[1] if y_true.ndim == 2 else 1
    scores = []
    if n_targets == 1:
        scores.append((names[0] if names else "target", mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)))
    else:
        for i in range(n_targets):
            scores.append((names[i] if names else f"y{i}", mean_absolute_error(y_true[:, i], y_pred[:, i]), r2_score(y_true[:, i], y_pred[:, i])))
    return scores
