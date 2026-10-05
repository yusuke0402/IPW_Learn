import os
import datetime
import numpy as np
import pandas as pd
import yaml


class Result:
    @staticmethod
    def save_results(results, config):
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        scenario = config["scenario"]["data_scenario_id"]
        n_source = config["dataset"]["source_number"]

        output_dir = config.get("output_dir", "results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df = pd.DataFrame(results)
        # ファイル名にシナリオとNを含める
        base_name = f"result_{scenario}_N{n_source}_{now}"
        csv_path = f"{output_dir}/{base_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"詳細ログを保存しました: {csv_path}")

        # 統計量の計算
        estimates = df["estimate_value"]

        if not estimates.isnull().all():
            valid = df.dropna(subset=["estimate_value"])
            valid_estimates = valid["estimate_value"].values
            true_value = config["dataset"]["true_value"]

            def _coverage(lo_col, hi_col):
                if lo_col in valid.columns and hi_col in valid.columns:
                    lo = valid[lo_col].values
                    hi = valid[hi_col].values
                    return float(np.mean((lo <= true_value) & (true_value <= hi)))
                return None

            def _mean_col(col):
                if col in valid.columns:
                    return float(np.mean(valid[col].values))
                return None

            stats = {
                "mean": float(np.mean(valid_estimates)),
                "variance": float(np.var(valid_estimates, ddof=1)),
                "std_dev": float(np.std(valid_estimates, ddof=1)),
                "mse": float(np.mean((valid_estimates - true_value) ** 2)),
                "mean_bootstrap_variance": _mean_col("bootstrap_variance"),
                "mean_bootstrap_se": _mean_col("bootstrap_se"),
                "coverage_probability_95": _coverage(
                    "estimate_95ci_lower", "estimate_95ci_upper"
                ),
                "mean_ess_source": _mean_col("ess_source"),
                "mean_max_weight_source": _mean_col("max_weight_source"),
            }
        else:
            stats = {
                "mean": None,
                "variance": None,
                "std_dev": None,
                "mse": None,
                "mean_bootstrap_variance": None,
                "mean_bootstrap_se": None,
                "coverage_probability_95": None,
                "mean_ess_source": None,
                "mean_max_weight_source": None,
            }
        summary_data = {
            "method": "Inverse Propensityscore Weighted Learning",
            "weight_method": config["hyperparameters"].get("weight_method", "propensity"),
            "timestamp": now,
            "senario_name": scenario,
            "model_id": config["scenario"]["model_id"],
            "model_scenario_id": config["scenario"]["model_scenario_id"],
            "n_features": config["hyperparameters"]["n_features"],
            "target_number": config["dataset"]["target_number"],
            "source_number": n_source,
            "n_trial": config["hyperparameters"]["n_trial"],
            "statistics": stats,
            "notes": "Automated experiment run",
        }
        yaml_path = f"{output_dir}/summary_{scenario}_N{n_source}_{now}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                summary_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        print(f"統計量を保存しました: {yaml_path}")
