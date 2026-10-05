import os

# joblib ワーカーごとの BLAS 内部スレッドを抑止(IW-Learn と同じ)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp  # noqa: E402
import random  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from data import DataSets  # noqa: E402
from result import Result  # noqa: E402
from weight import (  # noqa: E402
    compute_weights,
    effective_sample_size,
    weighted_mean_difference,
)

# numpy 2.0.x + Apple Accelerate で matmul が出す誤検知の RuntimeWarning
# (結果は有限で正しい)を抑止する。値が本当に発散した場合は推定値が non-finite になり
# bootstrap では除外、点推定では summary の NaN として現れる。
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def _resolve_n_jobs(config):
    """ワーカー数。config 優先、"auto" なら CPU 数 - 2 を上限 8 でクリップ(IW-Learn と同じ)。"""
    requested = config["hyperparameters"].get("n_jobs", "auto")
    cpu = mp.cpu_count()
    if requested == "auto":
        return max(1, min(cpu - 2, 8))
    if isinstance(requested, int) and requested < 0:
        return max(1, cpu + 1 + requested)
    return max(1, int(requested))


def _weight_kwargs(config):
    """propensity 系のロジスティック回帰の罰則設定(weight.py の compute_weights へ渡す)。"""
    hp = config["hyperparameters"]
    return {
        "propensity_penalty": hp.get("propensity_penalty", "none"),
        "propensity_C": hp.get("propensity_C", 1.0),
    }


def _single_estimate(weight_method, target_x, target_y, source_x, source_y,
                     means, covariances, wkw):
    """重みの計算 → Hájek 型の重み付き平均差を 1 回行う。"""
    weight_target, weight_source = compute_weights(
        weight_method, target_x, source_x, means=means, covariances=covariances, **wkw
    )
    estimate = weighted_mean_difference(weight_target, target_y, weight_source, source_y)
    return estimate, weight_source


def _run_one_trial(i, config, weight_method, n_bootstraps, wkw):
    """trial i を独立に 1 つ実行する。joblib ワーカーから呼ばれる。"""
    random.seed(i)
    np.random.seed(i)

    # 1.データの生成(乱数消費順序は IW-Learn と同一)
    data = DataSets(config=config)
    data.generate_data()
    target_x, target_y = data.target_x, data.target_y
    source_x, source_y = data.source_x, data.source_y
    n_target, n_source = target_x.shape[0], source_x.shape[0]

    # 2.点推定
    estimate, weight_source = _single_estimate(
        weight_method, target_x, target_y, source_x, source_y,
        data.means, data.covariances, wkw,
    )

    # 3.bootstrap: target・source を再標本化し、重みの推定(傾向スコアの当てはめ、
    #   uLSIF の (σ, λ) 選択)を含むパイプライン全体を各反復で再実行する。
    rng = np.random.default_rng(i)
    boot_estimates = []
    for _ in range(n_bootstraps):
        t_idx = rng.choice(n_target, size=n_target, replace=True)
        s_idx = rng.choice(n_source, size=n_source, replace=True)
        try:
            b_est, _ = _single_estimate(
                weight_method,
                target_x[t_idx], target_y[t_idx],
                source_x[s_idx], source_y[s_idx],
                data.means, data.covariances, wkw,
            )
        except Exception:
            continue
        if np.isfinite(b_est):
            boot_estimates.append(b_est)

    row = {
        "trial": i + 1,
        "estimate_value": estimate,
        "ess_source": effective_sample_size(weight_source),
        "max_weight_source": float(np.max(weight_source)),
    }
    if n_bootstraps > 0 and len(boot_estimates) >= 2:
        boot_estimates = np.array(boot_estimates)
        boot_var = float(np.var(boot_estimates, ddof=1))
        row.update({
            "bootstrap_variance": boot_var,
            "bootstrap_se": float(np.sqrt(boot_var)),
            "estimate_95ci_lower": float(np.percentile(boot_estimates, 2.5)),
            "estimate_95ci_upper": float(np.percentile(boot_estimates, 97.5)),
            "bootstrap_n": int(len(boot_estimates)),
        })
    return row


def run_experiment(config):
    n_trial = config["hyperparameters"]["n_trial"]
    n_bootstraps = config["hyperparameters"].get("n_bootstraps", 1000)
    weight_method = config["hyperparameters"].get("weight_method", "propensity")
    wkw = _weight_kwargs(config)
    n_jobs = _resolve_n_jobs(config)

    print(f"並列実行: n_jobs={n_jobs} (CPU={mp.cpu_count()}), "
          f"weight_method={weight_method}, n_bootstraps={n_bootstraps}, "
          f"propensity_penalty={wkw['propensity_penalty']}")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_run_one_trial)(i, config, weight_method, n_bootstraps, wkw)
        for i in range(n_trial)
    )

    # 4.結果の保存
    Result.save_results(results, config)


if __name__ == "__main__":
    # macOS の spawn 起動方式に対応するため main 保護は必須。
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_experiment(config)
