import numpy as np
import yaml
from data import DataSets
from trim import target_trim,source_trim
from propensityscore import propensityscore
from calculate_overlapping import estimate_r
from calculate_weight import estimate_weight
from composite_likelihood import etstimate_theta

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
theta_hat=np.empty(100)
split=config["hyperparam"]["n_split"]
for j in range(0,100):
    #結果の置き場
    est_1_r=np.empty(split)
    est_1_lamda=np.empty(split)
    N_1_k=np.empty(split)
    est_1_theta=np.empty(split)
    est_1_sigma=np.empty(split)
    #データの作成
    data=DataSets()
    #傾向スコアの推定
    target_1_ppscore,source_1_ppscore=propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=data.training_historical_1_x[:,1:],source_y=data.training_historical_1_y)
    target_2_ppscore,source_2_ppscore=propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=data.training_historical_2_x[:,1:],source_y=data.training_historical_2_y)
    #傾向スコアにより、層化・トリミング
    target_1_data,split_1_values=target_trim(data.training_current_x,data.training_current_y.reshape(-1),target_1_ppscore)
    source_1_data=source_trim(data.training_historical_1_x,data.training_historical_1_y.reshape(-1),source_1_ppscore,split_1_values)
    target_2_data,split_2_values=target_trim(data.training_current_x,data.training_current_y.reshape(-1),target_2_ppscore)
    source_2_data=source_trim(data.training_historical_2_x,data.training_historical_2_y.reshape(-1),source_2_ppscore,split_2_values)

    
    A_1=config["datasettings"]["source1_borrow_number"]
    A_2=config["datasettings"]["source2_borrow_number"]
    A=A_1+A_2
    #r_k_jの推定
    for i in range(0,5):
        target_1_condition = target_1_data['Group'] == i+1
        source_1_condition = source_1_data['Group'] == i+1
        target_1_group=target_1_data[target_1_condition]
        source_1_group=source_1_data[source_1_condition]
        est_r_1_j=estimate_r(target_1_group['Propensity_Score'].to_numpy(),source_1_group['Propensity_Score'].to_numpy())
        N_1_k[i]=len(source_1_group)
        est_1_r[i]=est_r_1_j[0]
    sum_1_r=np.sum(est_1_r)    
    #λ_k_jの推定
    for i in range(0,5):
        est_1_lamda[i]=estimate_weight(sum_r=sum_1_r,r_k=est_1_r[i],A=A,N_j_k=N_1_k[i])
    #η_k_jの推定
   # for i in range(0,5):


    #θ_k_jの推定
    for i in range(0,5):
        target_1_condition=target_1_data['Group'] == i+1
        source_1_condition = source_1_data['Group'] == i+1
        target_1_group=target_1_data[target_1_condition]
        source_1_group=source_1_data[source_1_condition]
        est_1_theta[i] = etstimate_theta(target_1_group['Outcomes'].to_numpy(), source_1_group['Outcomes'].to_numpy(), est_1_lamda[i])

    #θ_jの推定
    theta_hat[j]=np.mean(est_1_theta)

    #κの推定

print(np.mean(theta_hat))
