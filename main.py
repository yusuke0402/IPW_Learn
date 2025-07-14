import numpy as np
import yaml
import random
from data import DataSets
from propensityscore import propensityscore
from bayesian_optimization import bayesian_optimization
from train import training


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
    weight=1/ppscore_source1
    #3.ベイズ最適化
    best_lambda=bayesian_optimization(data,weight,i)[0]
    #4.MSEの最小化
    best_coffience=training(data=data,pred=weight,lambda_=best_lambda)
    #5.推定値の計算
    result[i]=(np.mean(data.verifying_current_y)-np.mean(data.verifying_current_x@best_coffience))/config["hyperparam"]["alpha"]
    
    
    


print("thetaの平均値：",np.mean(result))
print("thetaのMSE：",(np.mean(result-true_values))**2)
print("thetaのbias：",np.mean(result-true_values))
print("thetaのvaruance：",np.var(result))
print("thetaのsd：",np.std(result))
print("thetaの推定値：",result)



