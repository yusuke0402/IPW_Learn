import os
import datetime
import pandas as pd
import yaml

from experiment_execution import summarize_trials
from experiment_diagnostics import DEFAULT_MIN_ESS, DEFAULT_MAX_BOOTSTRAP_FAILURE_RATE


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
        frame = df.rename(columns={
            "estimate_value": "method_estimate", "bootstrap_se": "method_boot_se",
            "estimate_95ci_lower": "method_ci_lower", "estimate_95ci_upper": "method_ci_upper",
            "ci_error": "method_ci_error", "boot_n": "method_boot_n"})
        frame = frame.rename(columns={c: "method_" + c for c in (
            "point_error_type", "boot_requested", "boot_failed", "boot_failed_ESS", "ess",
            "min_ess", "max_bootstrap_failure_rate")})
        stats = summarize_trials(frame, "method", config["dataset"]["true_value"])
        stats.update(std_dev=stats["sd"], mse=stats["rmse"] ** 2,
                     mean_bootstrap_variance=float(df["bootstrap_variance"].mean()),
                     mean_bootstrap_se=float(df["bootstrap_se"].mean()),
                     coverage_probability_95=stats["coverage"])
        stats.update(mean_ess_source=float(df["ess_source"].mean()),
                     mean_max_weight_source=float(df["max_weight_source"].mean()))
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
            "feasibility_rule": {
                "min_ess": config["hyperparameters"].get("min_ess", DEFAULT_MIN_ESS),
                "max_bootstrap_failure_rate": config["hyperparameters"].get(
                    "max_bootstrap_failure_rate", DEFAULT_MAX_BOOTSTRAP_FAILURE_RATE),
                "scope": "historical sample in point and each full-pipeline bootstrap fit",
            },
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
