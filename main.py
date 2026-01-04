import numpy as np
import pandas as pd
import yaml
import random
import os
import datetime
from data import DataSets
from propensityscore import propensityscore
from wasserstain_distance import wasserstein_distance



#0.初期設定
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
split=config["experiment"]["hyperparameters"]["n_split"]
results=[]
source_list = config["experiment"]["dataset"]["source"]

for i in range(0,config["experiment"]["hyperparameters"]["n_trial"]):
    #1.データ作成
    np.random.seed(i)
    random.seed(i)
    
    data=DataSets()

    #2.傾向スコアの推定
    ppscore_target_1,ppscore_source1 = propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=data.training_historical_1_x[:,1:],source_y=data.training_historical_1_y)
    ppscore_target_2,ppscore_source2 = propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=data.training_historical_2_x[:,1:],source_y=data.training_historical_2_y)
    
    #3.wasserstain_distanceの計算
    wasserstein_dist_1 = wasserstein_distance(data.training_current_x[:,1:], data.training_historical_1_x[:,1:])
    wasserstein_dist_2 = wasserstein_distance(data.training_current_x[:,1:], data.training_historical_2_x[:,1:])
    #4.重みの計算
    weight_source_1=1/(1-ppscore_source1)
    weight_source_2=1/(1-ppscore_source2)
    weight_target_1=1/(ppscore_target_1)  
    weight_target_2=1/(ppscore_target_2)
    weight_mean_1=wasserstein_dist_2/(wasserstein_dist_1+wasserstein_dist_2)
    weight_mean_2=wasserstein_dist_1/(wasserstein_dist_1+wasserstein_dist_2)
    #5.推定値の計算
    mean_1= np.sum(weight_target_1 * data.training_current_y.ravel())/np.sum(weight_target_1)-np.sum(weight_source_1 * data.training_historical_1_y.ravel()) / np.sum(weight_source_1)
    mean_2= np.sum(weight_target_2 * data.training_current_y.ravel())/np.sum(weight_target_2)-np.sum(weight_source_2 * data.training_historical_2_y.ravel()) / np.sum(weight_source_2)
    estimate= mean_1 * weight_mean_1 + mean_2 * weight_mean_2
    
    results.append({
        "trial" : i+1,
        "estimate_value" : estimate,
        "weight_source_1" :weight_mean_1,
        "weight_source_2" : weight_mean_2,
        "source_1_mean" : mean_1,
        "source_2_mean" : mean_2
    
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
    "mse": float(np.mean((estimates - config["experiment"]["hyperparameters"]["true_params"]["value"]) ** 2)),
}

summary_data = {
    "timestamp": now,
    "senario_name": config["experiment"]["scenario"]["pattern_name"],
    "current_number": config["experiment"]["dataset"]["target_number"],
    "historical_1_number": next(s["number"] for s in source_list if s["name"] == "source1"),
    "historical_2_number": next(s["number"] for s in source_list if s["name"] == "source2"),
    "n_trial": config["experiment"]["hyperparameters"]["n_trial"],
    "statistics": stats,
    "notes": "YOU CAN WRITE SOME NOTES HERE",
}
with open(f"results/summary_{now}.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(
        summary_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
    )
print(f"統計量を保存しました: results/summary_{now}.yaml")
    
    
    





