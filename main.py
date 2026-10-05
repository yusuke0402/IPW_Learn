import random

import numpy as np
import yaml

from data import DataSets
from result import Result
from weight import compute_weights, effective_sample_size, weighted_mean_difference


def _weight_kwargs(config):
    hp = config["hyperparameters"]
    return {
        "propensity_penalty": hp.get("propensity_penalty", "none"),
        "propensity_C": hp.get("propensity_C", 1.0),
    }


def _estimate(weight_method, target_x, target_y, source_x, source_y, means, covariances, wkw):
    """重みの計算(傾向スコア / uLSIF / oracle / 一様)→ Hájek 型の重み付き平均差。"""
    weight_target, weight_source = compute_weights(
        weight_method, target_x, source_x, means=means, covariances=covariances, **wkw
    )
    estimate = weighted_mean_difference(weight_target, target_y, weight_source, source_y)
    return estimate, weight_source


def run_experiment(config):
    n_trial = config["hyperparameters"]["n_trial"]
    # 0 なら点推定のみ。>0 なら target・source を復元抽出し、重みも反復ごとに推定し直す
    n_bootstraps = int(config["hyperparameters"].get("n_bootstraps", 0))
    weight_method = config["hyperparameters"].get("weight_method", "propensity")
    wkw = _weight_kwargs(config)
    results = []

    for i in range(0, n_trial):
        np.random.seed(i)
        random.seed(i)
        # 1.データの生成
        data = DataSets(config=config)
        data.generate_data()
        target_x, target_y = data.target_x, data.target_y
        source_x, source_y = data.source_x, data.source_y

        # 2-3. 重みの計算と推定
        estimate, weight_source = _estimate(
            weight_method, target_x, target_y, source_x, source_y,
            data.means, data.covariances, wkw,
        )
        row = {
            "trial": i + 1,
            "estimate_value": estimate,
            "ess_source": effective_sample_size(weight_source),
            "max_weight_source": float(np.max(weight_source)),
        }

        # 4. full-pipeline bootstrap(任意)
        if n_bootstraps > 0:
            rng = np.random.default_rng(i)
            n_t, n_s = target_x.shape[0], source_x.shape[0]
            boot = []
            for _ in range(n_bootstraps):
                t_idx = rng.choice(n_t, size=n_t, replace=True)
                s_idx = rng.choice(n_s, size=n_s, replace=True)
                try:
                    b, _ = _estimate(
                        weight_method, target_x[t_idx], target_y[t_idx],
                        source_x[s_idx], source_y[s_idx],
                        data.means, data.covariances, wkw,
                    )
                    if np.isfinite(b):
                        boot.append(b)
                except Exception:
                    continue
            boot = np.array(boot)
            if boot.size >= 2:
                row.update({
                    "bootstrap_variance": float(np.var(boot, ddof=1)),
                    "bootstrap_se": float(np.std(boot, ddof=1)),
                    "estimate_95ci_lower": float(np.percentile(boot, 2.5)),
                    "estimate_95ci_upper": float(np.percentile(boot, 97.5)),
                    "boot_n": int(boot.size),
                })
        results.append(row)

    # 5.結果の保存
    Result.save_results(results, config)


if __name__ == "__main__":
    # 直接実行された場合のデフォルト挙動
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_experiment(config)
