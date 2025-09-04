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
weight_mean_1_list=np.empty(config["hyperparam"]["n_trial"])
weight_mean_2_list=np.empty(config["hyperparam"]["n_trial"])
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
    weight_source_1=1/(1-ppscore_source1)
    weight_source_2=1/(1-ppscore_source2)
    weight_target_1=1/(ppscore_target_1)  
    weight_target_2=1/(ppscore_target_2)
    weight_mean_1=wasserstein_dist_2/(wasserstein_dist_1+wasserstein_dist_2)
    weight_mean_2=wasserstein_dist_1/(wasserstein_dist_1+wasserstein_dist_2)
    #5.推定値の計算
    mean_1= np.sum(weight_target_1 * data.training_current_y.ravel())/np.sum(weight_target_1)-np.sum(weight_source_1 * data.training_historical_1_y.ravel()) / np.sum(weight_source_1)
    mean_2= np.sum(weight_target_2 * data.training_current_y.ravel())/np.sum(weight_target_2)-np.sum(weight_source_2 * data.training_historical_2_y.ravel()) / np.sum(weight_source_2)
    ate= mean_1 * weight_mean_1 + mean_2 * weight_mean_2
    result[i]=ate
    weight_mean_1_list[i]=weight_mean_1
    weight_mean_2_list[i]=weight_mean_2

with open("result.txt", "a", encoding="utf-8") as f:
    print("==================================", file=f)
    print("trail：",config["senario"]["pattern_id"], file=f)
    print("n_target：",config["datasettings"]["target_number"], file=f)
    print("n_source1：",config["datasettings"]["source1_number"], file=f)
    print("n_source2：",config["datasettings"]["source2_number"], file=f)
    print("mean：", np.mean(result), file=f)
    print("MSE：", np.mean((result-true_values)**2), file=f)
    print("bias：", np.mean(result-true_values), file=f)
    print("variance：", np.var(result), file=f)
    print("sd：", np.std(result), file=f)
    print("weight_mean_1:", weight_mean_1_list, file=f)
    print("weight_mean_2:", weight_mean_2_list, file=f)









