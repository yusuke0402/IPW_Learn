import random

import numpy as np
import yaml

from data import DataSets
from result import Result
from weight import compute_weights, effective_sample_size, weighted_mean_difference


def run_experiment(config):
    n_trial = config["hyperparameters"]["n_trial"]
    weight_method = config["hyperparameters"].get("weight_method", "propensity")
    results = []

    for i in range(0, n_trial):
        np.random.seed(i)
        random.seed(i)
        # 1.データの生成
        data = DataSets(config=config)
        data.generate_data()
        target_x, target_y = data.target_x, data.target_y
        source_x, source_y = data.source_x, data.source_y

        # 2.重みの計算(傾向スコア / uLSIF / oracle / 一様)
        weight_target, weight_source = compute_weights(
            weight_method,
            target_x,
            source_x,
            means=data.means,
            covariances=data.covariances,
        )

        # 3.推定値の計算(Hájek 型の重み付き平均差)
        estimate = weighted_mean_difference(
            weight_target, target_y, weight_source, source_y
        )
        results.append(
            {
                "trial": i + 1,
                "estimate_value": estimate,
                "ess_source": effective_sample_size(weight_source),
                "max_weight_source": float(np.max(weight_source)),
            }
        )

    # 4.結果の保存
    Result.save_results(results, config)


if __name__ == "__main__":
    # 直接実行された場合のデフォルト挙動
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_experiment(config)
