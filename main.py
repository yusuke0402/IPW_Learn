import numpy as np
import pandas as pd
import yaml
import random
import os
import datetime
from data import DataSets
from propensityscore import propensityscore



#0.初期設定
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
split=config["hyperparameters"]["n_split"]
results=[]

for i in range(0,config["hyperparameters"]["n_trial"]):
    #1.データ作成
    np.random.seed(i)
    random.seed(i)
    
    data=DataSets()

    #2.傾向スコアの推定
    ppscore_target,ppscore_source = propensityscore(target_x=data.training_target_x[:,1:],target_y=data.training_target_y,source_x=data.training_source_x[:,1:],source_y=data.training_source_y)
    
    #4.重みの計算
    weight_source=1/(1-ppscore_source)
    weight_target=1/(ppscore_target)
    #5.推定値の計算
    mean= np.sum(weight_target * data.training_target_y.ravel())/np.sum(weight_target)-np.sum(weight_source * data.training_source_y.ravel()) / np.sum(weight_source)
    estimate= mean
    results.append({
        "trial" : i+1,
        "estimate_value" : estimate
    })

# 結果の保存
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
    
    
    





