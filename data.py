import numpy as np
import pandas as pd
import yaml
from model_factory import ModelFactory
class DataSets:

  with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

  target_split = int(config["hyperparameters"]["split_ratio"] * config["dataset"]["target_number"])

 #初期化
  def __init__(self):    
    data_scenario_id=DataSets.config["scenario"]["data_scenario_id"]
    n_features= DataSets.config["hyperparameters"]["n_features"]
    mean_df=pd.read_csv(f"configs/{n_features}dim_means.csv")
    cov_df=pd.read_csv(f"configs/{n_features}dim_covariances.csv")
    

    def generate_multivariate_normal(domain, size=1):
      mean_vector = means[domain]  # 平均ベクトルを1次元配列として生成
      cov_matrix = covariances[domain] #分散共分散行列
      x=np.random.multivariate_normal(mean_vector, cov_matrix, size)
      return np.insert(x,0,1,axis=1),mean_vector

    means={
    row['domain']: np.array([row[f'dim{i+1}'] for i in range(n_features)])
      for _, row in mean_df.iterrows() if row['data_scenario_id']==data_scenario_id
  } 
    covariances={}
    for _, row in cov_df.iterrows():
      if row['data_scenario_id']!=data_scenario_id:
        continue
      flat=np.array([float(x) for x in row['cov'].split(',')])
      cov=flat.reshape((n_features,n_features))
      covariances[row['domain']]=cov


    self.__target_number=DataSets.config["dataset"]["target_number"] #現在試験の被験者数
    self.__target_x, self.__target_x_mean=generate_multivariate_normal("target",size=self.__target_number) #現在試験の共変量　行が被験者、列が共変量
    self.__epsilon=np.random.normal(loc=0,scale=1,size=self.__target_number) #測定誤差
    self.__target_y=ModelFactory.get_output(
            input=self.__target_x, t=np.ones(self.__target_number), epsilon=self.__epsilon
    )

    self.__source_number = DataSets.config["dataset"]["source_number"]
    self.__source_x, self.__source_x_mean=generate_multivariate_normal("source",size=self.__source_number)
    self.__epsilon=np.random.normal(loc=0,scale=1,size=self.__source_number)
    self.__source_y=ModelFactory.get_output(
            input=self.__source_x, t=np.zeros(self.__source_number), epsilon=self.__epsilon
        )
  
#変数のカプセル化、意図しない値の書き換えを防ぐ目的
  @property
  def training_target_x(self):
     return self.__target_x[0:self.target_split,:]
  @property
  def verifying_target_x(self):
     return self.__target_x[self.target_split:,:]
  @property
  def training_target_y(self):
     return self.__target_y[0:self.target_split,:]
  @property
  def verifying_target_y(self):
     return self.__target_y[self.target_split:,:]
  @property
  def training_source_x(self):
    return self.__source_x
  @property
  def training_source_y(self):
    return self.__source_y
  @property
  def current_x_mean(self):
    return self.__target_x_mean
  @property
  def source_x_mean(self):
    return self.__source_x_mean