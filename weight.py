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
                     L2の強さCは所属群の層化CVで選ぶ。CV無効時は propensity_C、
                     propensity_penalty="none" のときは罰則なしの最尤推定を使う。
- "ulsif"          : uLSIF(Kanamori et al. 2009)で r(x) を直接推定。元観測単位の
                     CVで (σ, λ) を選び、各foldの基底は学習側targetだけから選ぶ。
- "oracle"         : データ生成に使った真のガウス分布から r(x) を解析的に計算。

x は先頭列に切片 1 を持つ (n, 1 + n_features) の設計行列を想定する。
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

from propensityscore import propensityscore
from oracle_weight import _gaussian_logpdf as _gaussian_logpdf
from oracle_weight import gaussian_oracle_metadata, oracle_density_ratio
from experiment_diagnostics import (
    DEFAULT_MIN_ESS, NUMERICAL_ERRORS, assess_ess, validate_min_ess, weight_diagnostics,
)

WEIGHT_METHODS = ("uniform", "propensity", "ulsif", "oracle")


def compute_weights(method, target_x, source_x, means=None, covariances=None,
                    propensity_penalty="l2", propensity_C=1.0,
                    min_ess=DEFAULT_MIN_ESS, diagnostics=None,
                    propensity_cv=True, propensity_C_grid=None, propensity_cv_folds=5, bootstrap=False,
                    target_ids=None, source_ids=None):
    """(w_target, w_source) を返す。どちらも 1 次元配列。

    propensity_penalty / propensity_C は "propensity" のときの
    ロジスティック回帰の罰則("l2" = sklearn 既定の L2(既定)、"none" = 最尤)。
    """
    min_ess = validate_min_ess(min_ess)
    ps_selection = {}
    ps_kw = dict(penalty=propensity_penalty, C=propensity_C, cv=propensity_cv,
                 C_grid=propensity_C_grid, cv_folds=propensity_cv_folds, diagnostics=ps_selection)
    if method == "uniform":
        return np.ones(target_x.shape[0]), np.ones(source_x.shape[0])
    if method == "propensity":
        result = _propensity_odds_weights(target_x, source_x, **ps_kw)
        if diagnostics is not None:
            diagnostics["propensity_selection"] = ps_selection
        return result
    if method == "ulsif":
        model = ULSIF(min_ess=min_ess).fit(
            target_x, source_x, bootstrap=bootstrap, target_ids=target_ids, source_ids=source_ids)
        weights = model.density_ratio(source_x)
        if diagnostics is not None:
            diagnostics.update(raw=model.raw_weight_diagnostics_,
                               zero_weight_fallback=model.ratio_zero_fallback_,
                               ratio_sigma=model.sigma_, ratio_lambda=model.lambda_,
                               ratio_cv_policy=model.CV_POLICY)
        return np.ones(target_x.shape[0]), weights
    if method == "oracle":
        if means is None or covariances is None:
            raise ValueError("oracle weights need the true means and covariances")
        weights = oracle_density_ratio(source_x[:, 1:], means, covariances)
        if diagnostics is not None:
            diagnostics["oracle_distribution"] = gaussian_oracle_metadata(means, covariances)
        return np.ones(target_x.shape[0]), weights
    raise ValueError(f"Unknown weight method: {method!r} (choose from {WEIGHT_METHODS})")


def weighted_mean_difference(weight_target, target_y, weight_source, source_y, min_ess=DEFAULT_MIN_ESS,
                             *, bootstrap=False, min_positive_weights=1):
    """Hájek 型の重み付き平均差。"""
    assess_ess(weight_source, min_ess, bootstrap=bootstrap, min_positive_weights=min_positive_weights)
    if weight_diagnostics(weight_target)["positive_weights"] == 0:
        raise ValueError("Target weights must have positive sum.")
    target_y = np.asarray(target_y).ravel()
    source_y = np.asarray(source_y).ravel()

    def mean(weights, values):
        # The common scale cancels in the Hájek mean; keep returned raw weights intact.
        w = np.asarray(weights, dtype=float).ravel()
        scaled = w / w.max()
        probabilities = scaled / scaled.sum()
        return float(np.dot(probabilities, values))

    return mean(weight_target, target_y) - mean(weight_source, source_y)


def effective_sample_size(weight):
    """Kish の有効標本数 (Σw)^2 / Σw^2。"""
    return weight_diagnostics(weight)["ess"]


def ipw_estimate(target_x, target_y, source_x, source_y, method="propensity",
                 min_ess=DEFAULT_MIN_ESS, bootstrap=False,
                 target_ids=None, source_ids=None, **kwargs):
    """Compute source ESS before taking any weighted mean."""
    minimum = validate_min_ess(min_ess)
    metadata, diagnostics = {}, {}
    try:
        wt, ws = compute_weights(method, target_x, source_x, min_ess=minimum,
                                 diagnostics=metadata, bootstrap=bootstrap,
                                 target_ids=target_ids, source_ids=source_ids, **kwargs)
        ps_selection = metadata.pop("propensity_selection", {})
        diagnostics = assess_ess(ws, minimum, bootstrap=bootstrap,
            min_positive_weights=source_x.shape[1], propensity_C=ps_selection.get("C"), **metadata)
        estimate = weighted_mean_difference(wt, target_y, ws, source_y, min_ess=minimum,
            bootstrap=bootstrap, min_positive_weights=source_x.shape[1])
        return dict(estimate=estimate, propensity_selection=ps_selection, ess=diagnostics["full"]["ess"],
                    max_weight=float(np.max(ws)), source_weights=ws,
                    ess_diagnostics=diagnostics)
    except NUMERICAL_ERRORS as exc:
        if not hasattr(exc, "ess_diagnostics"):
            exc.ess_diagnostics = diagnostics
        raise


# ---------------------------------------------------------------- propensity
def _propensity_odds_weights(target_x, source_x, **ps_kw):
    # propensityscore は P(R=1|x)=P(target|x) を返す(切片列は除いて渡す)
    _, ps_source = propensityscore(target_x[:, 1:], source_x[:, 1:], **ps_kw)
    ps_source = np.clip(ps_source, 1e-12, 1 - 1e-12)
    return np.ones(target_x.shape[0]), ps_source / (1.0 - ps_source)


# --------------------------------------------------------------------- uLSIF
class ULSIF:
    """元観測単位のCVと学習fold内の基底を使うuLSIF密度比推定。"""

    SIGMA_ANCHOR = "all_off_diagonal_distances"
    CV_POLICY = "observation_groups_train_target_basis_v1"
    SIGMA_FACTORS = (0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0)
    LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
    CV_FOLDS = 5
    CV_SEED = 12345
    MAX_BASIS = 100  # Kanamori et al. 2009 §2.5: b = min(100, n_te)

    def __init__(self, min_ess=DEFAULT_MIN_ESS):
        self.min_ess = validate_min_ess(min_ess)

    def fit(self, target_x, source_x, *, bootstrap=False, target_ids=None, source_ids=None):
        self.bootstrap_ = bool(bootstrap)
        n_source = source_x.shape[0]
        sigma_med = self._median_pairwise_distance(np.vstack((target_x, source_x)))

        # 最終学習は全target、CVは各学習foldから別々に基底を選ぶ。
        x_basis = self._select_basis(target_x)
        sigma, lambda_reg = self._select_hyperparams(
            target_x, source_x, x_basis, sigma_med, target_ids=target_ids, source_ids=source_ids)

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
        self.raw_weight_diagnostics_ = weight_diagnostics(r)
        self.ratio_zero_fallback_ = self.min_ess == 0 and not getattr(self, "bootstrap_", False) and self.raw_weight_diagnostics_["positive_weights"] == 0
        if self.ratio_zero_fallback_:
            return np.ones(x.shape[0])
        return r

    def _select_basis(self, target_x, rng=None):
        if len(target_x) <= self.MAX_BASIS:
            return target_x
        generator = np.random if rng is None else rng
        idx = generator.choice(len(target_x), size=self.MAX_BASIS, replace=False)
        return target_x[np.sort(idx)]

    @staticmethod
    def _observation_groups(x, ids):
        if ids is None:
            # IDを持たない配列呼出しでは同一X行を保守的に同じgroupとする。
            return np.unique(x, axis=0, return_inverse=True)[1]
        ids = np.asarray(ids)
        if ids.ndim != 1 or len(ids) != len(x):
            raise ValueError("Observation IDs must be a vector with one ID per row.")
        if ids.dtype.kind in "fc" and not np.isfinite(ids).all():
            raise ValueError("Observation IDs must be finite.")
        return np.unique(ids, return_inverse=True)[1]

    def _make_cv_folds(self, target_x, source_x, *, target_ids=None, source_ids=None):
        groups_t = self._observation_groups(target_x, target_ids)
        groups_s = self._observation_groups(source_x, source_ids)
        unique_t, unique_s = np.unique(groups_t), np.unique(groups_s)
        n_folds = min(self.CV_FOLDS, len(unique_t), len(unique_s))
        if n_folds < 2:
            raise ValueError("uLSIF CV requires at least two observation groups in each domain.")
        rng = np.random.default_rng(self.CV_SEED)
        folds_t = np.array_split(rng.permutation(unique_t), n_folds)
        folds_s = np.array_split(rng.permutation(unique_s), n_folds)
        folds = []
        for held_t, held_s in zip(folds_t, folds_s):
            valid_t = np.isin(groups_t, held_t)
            valid_s = np.isin(groups_s, held_s)
            folds.append((np.flatnonzero(~valid_t), np.flatnonzero(valid_t),
                          np.flatnonzero(~valid_s), np.flatnonzero(valid_s)))
        return folds

    def _select_hyperparams(self, target_x, source_x, x_basis, sigma_med, *,
                            target_ids=None, source_ids=None):
        """Grouped J-score CV。x_basis引数は既存呼出しとの互換性のため残す。"""
        folds = self._make_cv_folds(
            target_x, source_x, target_ids=target_ids, source_ids=source_ids)
        rng = np.random.default_rng(self.CV_SEED + 1)
        bases = [self._select_basis(target_x[tr_t], rng=rng) for tr_t, _, _, _ in folds]

        best = (np.inf, sigma_med, 1e-2)
        for fac in self.SIGMA_FACTORS:
            sig = fac * sigma_med
            gamma = 1.0 / (2 * sig**2)
            scores = np.zeros(len(self.LAMBDA_GRID))
            for (tr_t, va_t, tr_s, va_s), basis in zip(folds, bases):
                K_s = rbf_kernel(source_x, basis, gamma=gamma)
                K_t = rbf_kernel(target_x, basis, gamma=gamma)
                b = len(basis)
                H = K_s[tr_s].T @ K_s[tr_s] / len(tr_s)
                h = K_t[tr_t].mean(axis=0)
                for index, lam in enumerate(self.LAMBDA_GRID):
                    alpha = np.linalg.solve(H + lam * np.eye(b), h)
                    w_s = np.maximum(K_s[va_s] @ alpha, 0)
                    w_t = np.maximum(K_t[va_t] @ alpha, 0)
                    scores[index] += 0.5 * np.mean(w_s**2) - np.mean(w_t)
            for lam, score in zip(self.LAMBDA_GRID, scores / len(folds)):
                if score < best[0]:
                    best = (score, sig, lam)
        return best[1], best[2]

    @staticmethod
    def _median_pairwise_distance(x):
        dists = pairwise_distances(x, metric="euclidean")
        return np.percentile(dists[np.triu_indices_from(dists, k=1)], 50)
