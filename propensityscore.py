import numpy as np
from sklearn.linear_model import LogisticRegression


def propensityscore(target_x, source_x, penalty="l2", C=1.0, max_iter=1000):
    """target(R=1) と source(R=0) を結合してロジスティック回帰を当てはめ、
    各群の傾向スコア e(x) = P(R=1 | x) = P(target | x) を返す。

    penalty:
        "l2"   : sklearn 既定の L2 罰則(強さは C=1.0、小さいほど強い)。既定。
                 PSS-Learn の propensityscore.py と同じ推定量で、傾向スコア系の
                 手法間で揃えるためにこちらを既定にしている。
        "none" : 罰則なしの最尤推定。t50_comparison/run_ipw_t50.py の
                 ほぼ無罰則の Newton 法と同じ推定対象(感度解析用)。
    完全分離などで収束しない場合でも例外は投げず、sklearn の返す係数をそのまま使う
    (重みの極端さは ESS・最大重みの診断で確認する)。
    """
    merged_x = np.vstack([target_x, source_x])
    merged_r = np.concatenate(
        (np.ones(target_x.shape[0]), np.zeros(source_x.shape[0])), axis=0
    )

    if penalty == "none":
        model = LogisticRegression(penalty=None, max_iter=max_iter)
    elif penalty == "l2":
        model = LogisticRegression(penalty="l2", C=C, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown penalty: {penalty!r} (choose 'none' or 'l2')")
    model.fit(merged_x, merged_r)

    target_ps = model.predict_proba(target_x)[:, 1]
    source_ps = model.predict_proba(source_x)[:, 1]
    return target_ps, source_ps
