import numpy as np
import pandas as pd
import yaml
import random
import os
import datetime
from data import DataSets
from propensityscore import propensityscore
from result import Result



#0.初期設定
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
split=config["hyperparameters"]["n_split"]
results=[]
n_trial=config["hyperparameters"]["n_trial"]

for i in range(0,n_trial):
    np.random.seed(i)
    random.seed(i)
    #1.データの生成
    data=DataSets(config=config)
    data.generate_data()
    target_x=data.target_x
    source_x=data.source_x
    target_y=data.target_y
    source_y=data.source_y
    #2.傾向スコアの推定
    ppscore_target,ppscore_source = propensityscore(target_x=target_x[:,1:],target_y=target_y,source_x=source_x[:,1:],source_y=source_y)

    #3.重みの計算
    weight_source=1/(1-ppscore_source)
    weight_target=1/(ppscore_target)
    #4.推定値の計算
    mean= np.sum(weight_target * target_y.ravel())/np.sum(weight_target)-np.sum(weight_source * source_y.ravel()) / np.sum(weight_source)
    estimate= mean
    results.append({
        "trial" : i+1,
        "estimate_value" : estimate
    })

#5.結果の保存
    
Result.save_results(results, config)



