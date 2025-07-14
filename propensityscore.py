from sklearn.linear_model import LogisticRegression
import numpy as np
from data import DataSets

def propensityscore(target_x,target_y,source_x,source_y):

    marge_X=np.vstack([target_x,source_x])
    
    target_Y=np.zeros_like(target_y).reshape(-1)
    source1_Y=np.ones_like(source_y).reshape(-1)
    
    marge_Y=np.concatenate((target_Y,source1_Y),axis=0)
    
    logi_model=LogisticRegression()
    
    logi_model.fit(marge_X,marge_Y)
   
    source_propensityscore=logi_model.predict_proba(source_x)[:,0]
    target_propensityscore=logi_model.predict_proba(target_x)[:,0]
    
    return target_propensityscore,source_propensityscore


