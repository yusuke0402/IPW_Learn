from scipy.stats import gaussian_kde
from scipy.integrate import quad
import numpy as np
import pandas as pd

def estimate_r(ps_target,ps_source):
    #密度関数の推定
    kde_target=gaussian_kde(ps_target)
    kde_source=gaussian_kde(ps_source)
    #積分範囲の指定
    min_ps=min(ps_target.min(),ps_source.min())
    max_ps=max(ps_target.max(),ps_source.max())
    #被積分関数の定義
    def min_density(x):
        return np.minimum(kde_target(x), kde_source(x))
    #類似度rの計算
    est_r=quad(min_density,min_ps,max_ps)
    return est_r
