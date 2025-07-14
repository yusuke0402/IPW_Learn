import numpy as np
import pandas as pd
import yaml

class DataSets:

  with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

  target_split = int(config["hyperparam"]["split_ratio"] * config["datasettings"]["target_number"])

 #初期化
  def __init__(self):    
    pattern_id=DataSets.config["senario"]["pattern_id"]
    n_features=10
    mean_df=pd.read_csv(DataSets.config["path"]["mean_csv_path"])
    cov_df=pd.read_csv(DataSets.config["path"]["cov_csv_path"])
    

    def generate_multivariate_normal(domain, size=1):
      mean_vector = means[domain]  # 平均ベクトルを1次元配列として生成
      cov_matrix = covariances[domain] #分散共分散行列
      x=np.random.multivariate_normal(mean_vector, cov_matrix, size)
      return np.insert(x,0,1,axis=1),mean_vector

    def truefunction(x,t,epsilon):
      coefficient=np.array(DataSets.config["datasettings"]["true_coffience"])
      return np.array([coefficient@x.T+t+epsilon]).T


    means={
    row['domain']: np.array([row[f'dim{i+1}'] for i in range(n_features)])
      for _, row in mean_df.iterrows() if row['pattern_id']==pattern_id
  } 
    covariances={}
    for _, row in cov_df.iterrows():
      if row['pattern_id']!=pattern_id:
        continue
      flat=np.array([float(x) for x in row['cov'].split(',')])
      cov=flat.reshape((n_features,n_features))
      covariances[row['domain']]=cov


    self.__current_number=DataSets.config["datasettings"]["target_number"] #現在試験の被験者数
    self.__current_x, self.__current_x_mean=generate_multivariate_normal("A",size=self.__current_number) #現在試験の共変量　行が被験者、列が共変量
    # 2~5列目を0以上なら1、0未満なら0に変換
    self.__current_x[:,1:5] = (self.__current_x[:,1:5] >= 0).astype(int)
    self.__epsilon=np.random.normal(loc=0,scale=1,size=self.__current_number) #測定誤差
    self.__current_y=truefunction(x=self.__current_x,t=np.zeros(self.__current_number),epsilon=self.__epsilon).reshape(-1,1) #現在試験のアウトカム　列ベクトル

    self.__historical_1_number=DataSets.config["datasettings"]["source1_number"]
    self.__historical_1_x, self.__historical_1_x_mean=generate_multivariate_normal("B",size=self.__historical_1_number)
    # 2~5列目を0以上なら1、0未満なら0に変換
    self.__historical_1_x[:,1:5] = (self.__historical_1_x[:,1:5] >= 0).astype(int)
    self.__epsilon=np.random.normal(loc=0,scale=1,size=self.__historical_1_number)
    self.__historical_1_y=truefunction(x=self.__historical_1_x,t=np.zeros(self.__historical_1_number),epsilon=self.__epsilon).reshape(-1,1)

    self.__historical_2_number=DataSets.config["datasettings"]["source2_number"]
    self.__historical_2_x, self.__historical_2_x_mean=generate_multivariate_normal("C",size=self.__historical_2_number)
    # 2~5列目を0以上なら1、0未満なら0に変換
    self.__historical_2_x[:,1:5] = (self.__historical_2_x[:,1:5] >= 0).astype(int)
    self.__epsilon=np.random.normal(loc=0,scale=1,size=self.__historical_2_number)
    self.__historical_2_y=truefunction(x=self.__historical_2_x,t=np.zeros(self.__historical_2_number),epsilon=self.__epsilon).reshape(-1,1)

  
#変数のカプセル化、意図しない値の書き換えを防ぐ目的
  @property
  def training_current_x(self):
     return self.__current_x[0:self.target_split,:]
  @property
  def verifying_current_x(self):
     return self.__current_x[self.target_split:,:]
  @property
  def training_current_y(self):
     return self.__current_y[0:self.target_split,:]
  @property
  def verifying_current_y(self):
     return self.__current_y[self.target_split:,:]
  @property
  def training_historical_1_x(self):
    return self.__historical_1_x
  @property
  def training_historical_1_y(self):
    return self.__historical_1_y
  @property
  def training_historical_2_x(self):
    return self.__historical_2_x
  @property
  def training_historical_2_y(self):
    return self.__historical_2_y
  @property
  def current_x_mean(self):
    return self.__current_x_mean
  @property
  def historical_1_x_mean(self):
    return self.__historical_1_x_mean
  @property
  def historical_2_x_mean(self):
    return self.__historical_2_x_mean