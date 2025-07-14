import numpy as np
import pandas as pd

def target_trim(covariates,outcomes,propensity_scores):
    ids=np.arange(1,outcomes.shape[0]+1)
    data={'ID':ids,}
    for i in range(11):
        data[f'Covariate_{i+1}']=covariates[:,i]
    data['Outcomes']=outcomes
    data['Propensity_Score']=propensity_scores
    df=pd.DataFrame(data)
    df_sorted=df.sort_values(by='Propensity_Score',ascending=False).reset_index(drop=True)
    num_groups=5
    df_sorted['Group']=pd.qcut(df_sorted.index,q=num_groups,labels=False,duplicates='drop')+1
    max_min_values=df_sorted.groupby('Group')['Propensity_Score'].agg(['max','min'])
    return df_sorted,max_min_values

def source_trim(covariates,outcomes,propensity_scores,split_values):
    ids=np.arange(1,outcomes.shape[0]+1)
    data={'ID':ids,}
    for i in range(11):
        data[f'Covariate_{i+1}']=covariates[:,i]
    data['Outcomes']=outcomes
    data['Propensity_Score']=propensity_scores
    df=pd.DataFrame(data)
    max_value=split_values.loc[1,'max']
    min_value=split_values.loc[5,'min']
    condition_upper=df['Propensity_Score']<=max_value
    condition_lower=df['Propensity_Score']>=min_value
    filtered_condition=condition_lower&condition_upper
    df_filtered=df[filtered_condition]
    df_sorted=df_filtered.sort_values(by='Propensity_Score',ascending=False).reset_index(drop=True)
    bins=np.array([split_values.loc[5,'min'],split_values.loc[5,"max"],split_values.loc[4,"max"],split_values.loc[3,"max"],split_values.loc[2,"max"],split_values.loc[1,"max"]])
    source_group_number=pd.cut(df_sorted['Propensity_Score'],bins=bins,labels=False,include_lowest=True)+1
    num_groups = len(bins) - 1
    df_sorted['Group'] = (num_groups + 1) - source_group_number
    return df_sorted