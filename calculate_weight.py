from scipy.stats import gaussian_kde
from scipy.integrate import quad
import numpy as np
import pandas as pd

def estimate_weight(sum_r,r_k,A,N_j_k):
    tmp=(A*r_k)/sum_r
    return min(tmp,N_j_k)