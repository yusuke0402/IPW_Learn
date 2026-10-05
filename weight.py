"""source(過去対照群)を target(現在試験)の共変量分布へ輸送する重みの計算。

推定対象は現在試験集団における治療効果 Δ_c = E[Y(1) - Y(0) | R=1]。
target 側は重みなし(w_t = 1)、source 側に密度比
    r(x) = p_target(x) / p_source(x)
に比例する重み w_s を掛け、Hájek 型(自己正規化)で平均差を取る:

    Δ̂ = Σ_i w_t,i y_t,i / Σ_i w_t,i − Σ_j w_s,j y_s,j / Σ_j w_s,j

`method` で重みの種類を選ぶ(configs/config.yaml の hyperparameters.weight_method):

- "uniform"        : w_s = 1。輸送なしの単純平均差(ベースライン)。
- "propensity"     : ロジスティック回帰の傾向スコア e(x)=P(R=1|x) のオッズ
                     w_s = e/(1-e)。密度比 r(x) に比例する(定数倍は自己正規化で消える)。
                     罰則は hyperparameters.propensity_penalty("l2" = sklearn 既定、
                     強さ propensity_C、既定 / "none" = 最尤)で選ぶ。
- "propensity_ate" : 旧実装。w_t = 1/e, w_s = 1/(1-e) で target/source の両方を
                     混合集団へ重み付けする(推定対象が Δ_c ではない点に注意)。
- "ulsif"          : uLSIF(Kanamori et al. 2009)で r(x) を直接推定。IW-Learn の
                     importance_weighted_learn.py と同じ手順((σ, λ) を J 規準の
                     5-fold CV で選択、基底点は target から min(100, n_t) 点)。
- "oracle"         : データ生成に使った真のガウス分布から r(x) を解析的に計算。

x は先頭列に切片 1 を持つ (n, 1 + n_features) の設計行列を想定する。
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

from propensityscore import propensityscore

WEIGHT_METHODS = ("uniform", "propensity", "propensity_ate", "ulsif", "oracle")


def compute_weights(method, target_x, source_x, means=None, covariances=None,
                    propensity_penalty="l2", propensity_C=1.0):
    """(w_target, w_source) を返す。どちらも 1 次元配列。

    propensity_penalty / propensity_C は "propensity" / "propensity_ate" のときの
    ロジスティック回帰の罰則("l2" = sklearn 既定の L2(既定)、"none" = 最尤)。
    """
    ps_kw = {"penalty": propensity_penalty, "C": propensity_C}
    if method == "uniform":
        return np.ones(target_x.shape[0]), np.ones(source_x.shape[0])
    if method == "propensity":
        return _propensity_odds_weights(target_x, source_x, **ps_kw)
    if method == "propensity_ate":
        return _propensity_ate_weights(target_x, source_x, **ps_kw)
    if method == "ulsif":
        return np.ones(target_x.shape[0]), ULSIF().fit(target_x, source_x).density_ratio(source_x)
    if method == "oracle":
        if means is None or covariances is None:
            raise ValueError("oracle weights need the true means and covariances")
        return np.ones(target_x.shape[0]), oracle_density_ratio(
            source_x[:, 1:], means, covariances
        )
    raise ValueError(f"Unknown weight method: {method!r} (choose from {WEIGHT_METHODS})")


def weighted_mean_difference(weight_target, target_y, weight_source, source_y):
    """Hájek 型の重み付き平均差。"""
    target_y = np.asarray(target_y).ravel()
    source_y = np.asarray(source_y).ravel()
    return float(
        np.sum(weight_target * target_y) / np.sum(weight_target)
        - np.sum(weight_source * source_y) / np.sum(weight_source)
    )


def effective_sample_size(weight):
    """Kish の有効標本数 (Σw)^2 / Σw^2。"""
    weight = np.asarray(weight).ravel()
    return float(weight.sum() ** 2 / np.sum(weight**2))


# ---------------------------------------------------------------- propensity
def _propensity_odds_weights(target_x, source_x, **ps_kw):
    # propensityscore は P(R=1|x)=P(target|x) を返す(切片列は除いて渡す)
    _, ps_source = propensityscore(target_x[:, 1:], source_x[:, 1:], **ps_kw)
    ps_source = np.clip(ps_source, 1e-12, 1 - 1e-12)
    return np.ones(target_x.shape[0]), ps_source / (1.0 - ps_source)


def _propensity_ate_weights(target_x, source_x, **ps_kw):
    ps_target, ps_source = propensityscore(target_x[:, 1:], source_x[:, 1:], **ps_kw)
    ps_target = np.clip(ps_target, 1e-12, 1 - 1e-12)
    ps_source = np.clip(ps_source, 1e-12, 1 - 1e-12)
    return 1.0 / ps_target, 1.0 / (1.0 - ps_source)


# -------------------------------------------------------------------- oracle
def _gaussian_logpdf(x, mean, cov):
    d = x.shape[1]
    chol = np.linalg.cholesky(cov)
    diff = np.linalg.solve(chol, (x - mean).T)  # (d, n)
    maha = np.sum(diff**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (maha + logdet + d * np.log(2.0 * np.pi))


def oracle_density_ratio(x, means, covariances):
    """真のガウス分布に基づく r(x) = N(x; μ_t, Σ_t) / N(x; μ_s, Σ_s)。x は切片なし。"""
    log_r = _gaussian_logpdf(x, means["target"], covariances["target"]) - _gaussian_logpdf(
        x, means["source"], covariances["source"]
    )
    return np.exp(log_r)


# --------------------------------------------------------------------- uLSIF
class ULSIF:
    """IW-Learn の ImportaceWeightLearn._weight と同じ uLSIF 密度比推定。"""

    SIGMA_FACTORS = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0)
    LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
    CV_FOLDS = 5
    MAX_BASIS = 100  # Kanamori et al. 2009 §2.5: b = min(100, n_te)

    def fit(self, target_x, source_x):
        n_target = target_x.shape[0]
        n_source = source_x.shape[0]
        sigma_med = self._median_pairwise_distance(np.vstack((target_x, source_x)))

        # 基底点: target から非復元で min(100, n_t) 点(n_t <= 100 なら乱数を消費しない)
        if n_target > self.MAX_BASIS:
            idx = np.random.choice(n_target, size=self.MAX_BASIS, replace=False)
            x_basis = target_x[np.sort(idx)]
        else:
            x_basis = target_x

        sigma, lambda_reg = self._select_hyperparams(target_x, source_x, x_basis, sigma_med)

        gamma = 1.0 / (2 * sigma**2)
        K_s = rbf_kernel(source_x, x_basis, gamma=gamma)
        K_t = rbf_kernel(target_x, x_basis, gamma=gamma)
        H = K_s.T @ K_s / n_source + lambda_reg * np.eye(K_s.shape[1])
        h = K_t.mean(axis=0)
        self.coef_ = np.linalg.solve(H, h)
        self.basis_ = x_basis
        self.sigma_ = sigma
        self.lambda_ = lambda_reg
        return self

    def density_ratio(self, x):
        K = rbf_kernel(x, self.basis_, gamma=1.0 / (2 * self.sigma_**2))
        r = np.maximum(K @ self.coef_, 0.0)
        # 全重みゼロ(当てはめの退化)への保険: 一様重みに退避
        if r.sum() <= 0:
            return np.ones(x.shape[0])
        return r

    def _select_hyperparams(self, target_x, source_x, x_basis, sigma_med):
        """J 規準の 5-fold CV で (σ, λ) を選ぶ。fold 分割は固定シードでグローバル RNG を消費しない。"""
        rng = np.random.default_rng(12345)
        n_t, n_s = target_x.shape[0], source_x.shape[0]
        folds_t = np.array_split(rng.permutation(n_t), self.CV_FOLDS)
        folds_s = np.array_split(rng.permutation(n_s), self.CV_FOLDS)

        best = (np.inf, sigma_med, 1e-2)
        for fac in self.SIGMA_FACTORS:
            sig = fac * sigma_med
            gamma = 1.0 / (2 * sig**2)
            K_s = rbf_kernel(source_x, x_basis, gamma=gamma)
            K_t = rbf_kernel(target_x, x_basis, gamma=gamma)
            b = K_s.shape[1]
            for lam in self.LAMBDA_GRID:
                score = 0.0
                for k in range(self.CV_FOLDS):
                    tr_s = np.setdiff1d(np.arange(n_s), folds_s[k])
                    tr_t = np.setdiff1d(np.arange(n_t), folds_t[k])
                    H = K_s[tr_s].T @ K_s[tr_s] / len(tr_s)
                    h = K_t[tr_t].mean(axis=0)
                    alpha = np.linalg.solve(H + lam * np.eye(b), h)
                    w_s = np.maximum(K_s[folds_s[k]] @ alpha, 0)
                    w_t = np.maximum(K_t[folds_t[k]] @ alpha, 0)
                    score += 0.5 * np.mean(w_s**2) - np.mean(w_t)
                if score / self.CV_FOLDS < best[0]:
                    best = (score / self.CV_FOLDS, sig, lam)
        return best[1], best[2]

    @staticmethod
    def _median_pairwise_distance(x):
        dists = pairwise_distances(x, metric="euclidean")
        return np.percentile(dists[np.triu_indices_from(dists, k=1)], 50)
