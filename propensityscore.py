import numpy as np
from sklearn.linear_model import LogisticRegression


def propensityscore(target_x, source_x):
    """target(R=1) と source(R=0) を結合してロジスティック回帰を当てはめ、
    各群の傾向スコア e(x) = P(R=1 | x) = P(target | x) を返す。"""
    merged_x = np.vstack([target_x, source_x])
    merged_r = np.concatenate(
        (np.ones(target_x.shape[0]), np.zeros(source_x.shape[0])), axis=0
    )

    model = LogisticRegression()
    model.fit(merged_x, merged_r)

    target_ps = model.predict_proba(target_x)[:, 1]
    source_ps = model.predict_proba(source_x)[:, 1]
    return target_ps, source_ps
