import ot
import numpy as np

def wasserstein_distance(x,y):
      # コスト行列（ユークリッド距離）
      M = ot.dist(x, y)
      M /= M.max()
      # 一様分布の重み（経験分布
      a = np.ones((x.shape[0],)) / x.shape[0]
      b = np.ones((y.shape[0],)) / y.shape[0]
      # 1-Wasserstein距離（EMD）
      distance = ot.emd2(a, b, M) 
      return distance