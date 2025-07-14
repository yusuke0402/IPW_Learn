from skopt import gp_minimize
from skopt import forest_minimize
from skopt import gbrt_minimize
from skopt import dummy_minimize

import numpy as np
from train import training

#引数としてDataSet_classを、戻り値としてhyperparams(0:2がbeta,3がlambda)を返す
def bayesian_optimization(data,weitht,i):
    
    def objective(hyperparams):
        lambda_=hyperparams[0]
        best_coffience=training(data=data,pred=weitht,lambda_=lambda_)
        source_x=np.concatenate([data.training_historical_1_x,data.training_historical_2_x],0)
        source_y=np.concatenate([data.training_historical_1_y,data.training_historical_2_y],0).reshape(-1,1)
        tmp=np.mean((source_x@best_coffience-source_y)**2)
        return tmp

    gp_result=gp_minimize(objective,[(1e-2,10)],n_calls=200,random_state=i)
    #forest_result=forest_minimize(objective,[(1e-6,10),(1e-6,10),(1e-6,10),(0,100)],n_calls=100)
    #gbrf_result=gbrt_minimize(func=objective,dimensions=[(1e-6,10),(1e-6,10),(1e-6,10),(0,100)],n_calls=100)
    #dummy_result=dummy_minimize(objective,[(1e-6,10),(1e-6,10),(1e-6,10)],n_calls=200)
    return gp_result.x 