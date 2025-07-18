import numpy as np
from scipy.optimize import minimize  
import yaml
#引数にDataSets_class,重みをとり、戻り値として最適化された係数を返す
#configを参照して自動的にニューラルネットワークと線形モデルを使い分けたい
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
def training(data,pred,lambda_):

    source_x=np.concatenate([data.training_historical_1_x,data.training_historical_2_x],0)
    source_y=np.concatenate([data.training_historical_1_y,data.training_historical_2_y],0).reshape(-1,1)
    target_x=data.training_current_x
    target_y=data.training_current_y.reshape(-1,1)
    weight=pred
    theta_init=np.ones(source_x.shape[1])
    result=minimize(weighted_MSE,theta_init,args=(source_x,source_y,target_x,target_y,weight,lambda_),method="L-BFGS-B")
    return result.x

def weighted_MSE(theta,source_x,source_y,target_x,target_y,weight,lambda_):
    theta=np.array([theta]).T
    therm1=np.mean(config["hyperparam"]["alpha"]*weight@(source_x@theta-source_y)**2)
    therm2=np.mean((1-config["hyperparam"]["alpha"])*(target_x@theta-target_y)**2)
    therm3=lambda_* np.sum(np.abs(theta))
    return therm1+therm2+therm3

