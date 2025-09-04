import numpy as np
import yaml
import random
from data import DataSets
from propensityscore import propensityscore




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
    integrated_historical_x=np.concatenate((data.training_historical_1_x,data.training_historical_2_x),axis=0)
    integrated_historical_y=np.concatenate((data.training_historical_1_y,data.training_historical_2_y),axis=0)
    #2.傾向スコアの推定
    ppscore_target,ppscore_source = propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=integrated_historical_x[:,1:],source_y=integrated_historical_y)
    
    #3.重みの計算
    weight_source=1/(1-ppscore_source)
    weight_target=1/(ppscore_target)
    
    #4.推定値の計算
    result[i]=np.sum(weight_target*data.training_current_y.ravel()) /np.sum(weight_target)-np.mean(weight_source@integrated_historical_y.ravel())/np.sum(weight_source)

with open("result.txt", "a", encoding="utf-8") as f:
    print("==================================", file=f)
    print("trial:", config["senario"]["pattern_id"], file=f)
    print("n_target:", config["datasettings"]["target_number"], file=f)
    print("n_source1:", config["datasettings"]["source1_number"], file=f)   
    print("n_source2:", config["datasettings"]["source2_number"], file=f)
    print("mean：", np.mean(result), file=f)
    print("MSE：", np.mean((result-true_values)**2), file=f)
    print("bias：", np.mean(result-true_values), file=f)
    print("variance：", np.var(result), file=f)
    print("sd：", np.std(result), file=f)





