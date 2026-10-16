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
from result import Result
from experiment_execution import run_weighted_bootstrap, main_columns
from experiment_diagnostics import DEFAULT_MIN_ESS, DEFAULT_MAX_BOOTSTRAP_FAILURE_RATE  # noqa: E402
from weight import ipw_estimate


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
        "propensity_penalty": hp.get("propensity_penalty", "l2"),
        "propensity_C": hp.get("propensity_C", 1.0),
        "min_ess": hp.get("min_ess", DEFAULT_MIN_ESS),
    }


def _single_estimate(weight_method, target_x, target_y, source_x, source_y,
                     means, covariances, wkw):
    """重みの計算 → Hájek 型の重み付き平均差を 1 回行う。"""
    result = ipw_estimate(target_x, target_y, source_x, source_y,
                          method=weight_method, means=means, covariances=covariances, **wkw)
    return result["estimate"], result["source_weights"]


def _run_one_trial(i, config, weight_method, n_bootstraps, wkw):
    """trial i を独立に 1 つ実行する。joblib ワーカーから呼ばれる。"""
    random.seed(i)
    np.random.seed(i)

    # 1.データの生成(乱数消費順序は IW-Learn と同一)
    data = DataSets(config=config)
    data.generate_data()
    target_x, target_y = data.target_x, data.target_y
    source_x, source_y = data.source_x, data.source_y
    run = run_weighted_bootstrap(
        lambda tx, ty, sx, sy: ipw_estimate(
            tx, ty, sx, sy, method=weight_method, means=data.means,
            covariances=data.covariances, **wkw),
        target_x, target_y, source_x, source_y, n_bootstraps, i,
        min_ess=wkw.get("min_ess", DEFAULT_MIN_ESS),
        max_failure_rate=config["hyperparameters"].get(
            "max_bootstrap_failure_rate", DEFAULT_MAX_BOOTSTRAP_FAILURE_RATE))
    row = dict(trial=i + 1, **main_columns(run))
    row.update(ess_source=row["ess"], max_weight_source=run["point"].get("max_weight", np.nan))
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
