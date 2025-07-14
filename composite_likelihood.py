import numpy as np 
from scipy.stats import norm
from scipy.optimize import minimize

def etstimate_theta(y_current, y_rwd, lambda_kj):

    def negative_composite_log_likelihood(params, y_current, y_rwd, lambda_kj):
        theta, sigma = params
        if sigma <= 0:
             return np.inf
        # 現行データとRWDデータの対数尤度を計算
        log_lik_current = np.sum(norm.logpdf(y_current, loc=theta, scale=sigma))
        log_lik_rwd = np.sum(norm.logpdf(y_rwd, loc=theta, scale=sigma))
        N_kj = len(y_rwd)
        if N_kj == 0: # RWDの患者がいない場合は現行研究の尤度のみ
            composite_log_lik = log_lik_current
        else:
            # 重み係数 = λ / N
            discount_factor = lambda_kj / N_kj
            composite_log_lik = log_lik_current + discount_factor * log_lik_rwd

    # 最小化するため、負の値を返す
        return -composite_log_lik
    
    # 初期値の設定
    initial_params = [np.mean(y_current), np.std(y_current)]  

    # 最適化の実行
    result = minimize(negative_composite_log_likelihood, initial_params, args=(y_current, y_rwd, lambda_kj),
                      bounds=[(None, None), (1e-6, None)], method='L-BFGS-B')  

    # 最適化結果からパラメータを取得
    estimated_theta = result.x[0]
    return estimated_theta