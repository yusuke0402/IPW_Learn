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

        stats = {
            "mean": float(estimates.mean()) if not estimates.isnull().all() else None,
            "variance": float(estimates.var()) if not estimates.isnull().all() else None,
            "std_dev": float(estimates.std()) if not estimates.isnull().all() else None,
            "mse": float(np.mean((estimates - config["hyperparameters"]["true_value"]) ** 2)) if not estimates.isnull().all() else None,
        }
        summary_data = {
            "method": "Inverse Propensityscore Weighted Learning",
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
