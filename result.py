import os
import datetime
import numpy as np
import pandas as pd
import yaml

class Result:
    
    @staticmethod
    def save_results(results, config):
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if not os.path.exists("results"):
            os.makedirs("results")
        df = pd.DataFrame(results)
        df.to_csv(f"results/result_{now}.csv", index=False)
        print(f"詳細ログを保存しました: results/result_{now}.csv")
    # 統計量の計算
        estimates = df["estimate_value"]

        stats = {
            "mean": float(estimates.mean()),
            "variance": float(estimates.var()),
            "std_dev": float(estimates.std()),
            "mse": float(np.mean((estimates - config["hyperparameters"]["true_value"]) ** 2)),
        }   
        summary_data = {
            "timestamp": now,
            "senario_name": config["scenario"]["data_scenario_id"],
            "model_id":config["scenario"]["model_id"],
            "model_scenario_id":config["scenario"]["model_scenario_id"],
            "n_features":config["hyperparameters"]["n_features"],
            "target_number": config["dataset"]["target_number"],
            "source_number": config["dataset"]["source_number"],
            "n_trial": config["hyperparameters"]["n_trial"],
            "statistics": stats,
            "notes": "YOU CAN WRITE SOME NOTES HERE",
        }
        with open(f"results/summary_{now}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                summary_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )
        print(f"統計量を保存しました: results/summary_{now}.yaml")
    
    