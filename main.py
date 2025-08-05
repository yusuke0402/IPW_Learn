import numpy as np
import yaml
import random
from data import DataSets
from propensityscore import propensityscore
from wasserstain_distance import wasserstein_distance
from weight import weight_calculation



#0.初期設定
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
split=config["hyperparam"]["n_split"]
result=np.empty(config["hyperparam"]["n_trial"])
true_values=np.ones_like(result)
for i in range(0,config["hyperparam"]["n_trial"]):
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
    weight_1=wasserstein_dist_2*weight_calculation(ppscore_source1, wasserstein_dist_1, wasserstein_dist_2)
    weight_2=wasserstein_dist_1*weight_calculation(ppscore_source2, wasserstein_dist_1, wasserstein_dist_2)
    #5.推定値の計算
    result[i] = np.mean(data.training_current_y)-(weight_1@data.training_historical_1_y).item()-(weight_2@data.training_historical_2_y).item()
    print((weight_1@data.training_historical_1_y).item()) 
    print(np.mean(data.training_historical_1_y))
    print(wasserstein_dist_2/(wasserstein_dist_1 + wasserstein_dist_2))
    inv_ppscore = 1 / ppscore_source1
    sum_inv_ppscore = np.sum(inv_ppscore)
    print(np.sum(wasserstein_dist_2*inv_ppscore/sum_inv_ppscore/(wasserstein_dist_1 + wasserstein_dist_2)))
    print((inv_ppscore/sum_inv_ppscore)@data.training_historical_1_y)
    print(result[i])
    print(np.mean(data.training_current_y))
    print((weight_2@data.training_historical_2_y).item())
    
    
    





