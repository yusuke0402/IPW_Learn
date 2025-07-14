import numpy as np
import yaml
from data import DataSets
from propensityscore import propensityscore
from calculate_overlapping import estimate_r
from calculate_weight import estimate_weight
from composite_likelihood import etstimate_theta
from trim import target_trim,source_trim

#0.初期設定
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
A_1=config["datasettings"]["source1_borrow_number"]
A_2=config["datasettings"]["source2_borrow_number"]
A=A_1+A_2
split=config["hyperparam"]["n_split"]
est_theta=np.empty(config["hyperparam"]["n_trial"])
true_values=np.ones_like(est_theta)*9.36
for j in range(0,config["hyperparam"]["n_trial"]):
    #1.データ作成
    data=DataSets()

    est_r_1=np.empty(split)
    est_lamda_1=np.empty(split)
    est_eta_1=np.empty(split)
    N_1_k=np.empty(split)
    est_theta_1_k=np.empty(split)

    est_r_2=np.empty(split)
    est_lamda_2=np.empty(split)
    est_eta_2=np.empty(split)
    N_2_k=np.empty(split)
    est_theta_2_k=np.empty(split)

    #2.傾向スコアの推定
    ppscore_target_1,ppscore_source1 = propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=data.training_historical_1_x[:,1:],source_y=data.training_historical_1_y)
    ppscore_target_2,ppscore_source2 = propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=data.training_historical_2_x[:,1:],source_y=data.training_historical_2_y)

    #3.傾向スコアに基づいてデータを層別化・トリミング
    target_1_data,split_1_values=target_trim(data.training_current_x,data.training_current_y.reshape(-1),ppscore_target_1)
    source_1_data=source_trim(data.training_historical_1_x,data.training_historical_1_y.reshape(-1),ppscore_source1,split_1_values)
    target_2_data,split_2_values=target_trim(data.training_current_x,data.training_current_y.reshape(-1),ppscore_target_2)
    source_2_data=source_trim(data.training_historical_2_x,data.training_historical_2_y.reshape(-1),ppscore_source2,split_2_values)

    #4.r_j_kの推定
    for i in range(0,split):
        target_1_condition = target_1_data['Group'] == i+1
        source_1_condition = source_1_data['Group'] == i+1
        target_1_group=target_1_data[target_1_condition]
        source_1_group=source_1_data[source_1_condition]
        est_r_1_k=estimate_r(target_1_group['Propensity_Score'].to_numpy(),source_1_group['Propensity_Score'].to_numpy())
        N_1_k[i]=len(source_1_group)
        est_r_1[i]=est_r_1_k[0]
    sum_r_1=np.sum(est_r_1) 

    for i in range(0,split):
        target_2_condition = target_2_data['Group'] == i+1
        source_2_condition = source_2_data['Group'] == i+1
        target_2_group=target_2_data[target_2_condition]
        source_2_group=source_2_data[source_2_condition]
        est_r_2_k=estimate_r(target_2_group['Propensity_Score'].to_numpy(),source_2_group['Propensity_Score'].to_numpy())
        N_2_k[i]=len(source_2_group)
        est_r_2[i]=est_r_2_k[0]
    sum_r_2=np.sum(est_r_2) 

    #5.Λ_jの推定
    for i in range(0,split):
        est_eta_1[i]=estimate_weight(sum_r=sum_r_1,r_k=est_r_1[i],A=A_1,N_j_k=N_1_k[i])
    est_capital_lamda_1=np.sum(est_eta_1)

    for i in range(0,split):
        est_eta_2[i]=estimate_weight(sum_r=sum_r_2,r_k=est_r_2[i],A=A_2,N_j_k=N_2_k[i])
    est_capital_lamda_2=np.sum(est_eta_2)

    #6.R_jの推定
    est_capital_r_1=sum_r_1/split
    est_capital_r_2=sum_r_2/split

    #7.λ_jの推定
    for i in range(0,5):
        est_lamda_1[i]=estimate_weight(sum_r=sum_r_1,r_k=est_r_1[i],A=A,N_j_k=N_1_k[i])

    for i in range(0,5):
        est_lamda_2[i]=estimate_weight(sum_r=sum_r_2,r_k=est_r_2[i],A=A,N_j_k=N_2_k[i])

    #8.kappa_jの推定
    lamda1_r1_product=est_capital_lamda_1*est_capital_r_1
    lamda2_r2_product=est_capital_lamda_2*est_capital_r_2
    est_kappa_1=lamda1_r1_product/(lamda1_r1_product+lamda2_r2_product)
    est_kappa_2=lamda2_r2_product/(lamda1_r1_product+lamda2_r2_product)

    #9.θ_jの推定
    for i in range(0,5):
        target_1_condition=target_1_data['Group'] == i+1
        source_1_condition = source_1_data['Group'] == i+1
        target_1_group=target_1_data[target_1_condition]
        source_1_group=source_1_data[source_1_condition]
        est_theta_1_k[i] = etstimate_theta(target_1_group['Outcomes'].to_numpy(), source_1_group['Outcomes'].to_numpy(), est_lamda_1[i])
    est_theta_1=np.mean(est_theta_1_k)

    for i in range(0,5):
        target_2_condition=target_2_data['Group'] == i+1
        source_2_condition = source_2_data['Group'] == i+1
        target_2_group=target_2_data[target_2_condition]
        source_2_group=source_2_data[source_2_condition]
        est_theta_2_k[i] = etstimate_theta(target_2_group['Outcomes'].to_numpy(), source_2_group['Outcomes'].to_numpy(), est_lamda_2[i])
    est_theta_2=np.mean(est_theta_2_k)

    #10.^θの推定
    est_theta[j]=est_kappa_1*est_theta_1+est_kappa_2*est_theta_2

print("thetaの平均値：",np.mean(est_theta))
print("thetaのMSE：",(np.mean(est_theta-true_values))**2)
print("thetaのbias：",np.mean(est_theta-true_values))
print("thetaのvaruance：",np.var(est_theta))
print("thetaのsd：",np.std(est_theta))
print("thetaの推定値：",est_theta)



