import numpy as np

def weight_calculation(ppscore,wasserstein_dist_1, wasserstein_dist_2):
    inv_ppscore = 1 / ppscore
    sum_inv_ppscore = np.sum(inv_ppscore)
    return inv_ppscore/sum_inv_ppscore/(wasserstein_dist_1 + wasserstein_dist_2)