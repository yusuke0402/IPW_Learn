import numpy as np
import pandas as pd
import yaml

class DataSets:

  

 #初期化
  def __init__(self,config):    
    self.data_scenario_id=config["scenario"]["data_scenario_id"]
    self.n_features=config["hyperparameters"]["n_features"]
    self.mean_df=pd.read_csv(f"configs/{self.n_features}dim_means.csv")
    self.cov_df=pd.read_csv(f"configs/{self.n_features}dim_covariances.csv")
    self.coefficient_df=pd.read_csv("configs/coefficient.csv")
    self.__load_means_and_covariances()
    self.config=config
    
  def __generate_multivariate_normal(self, domain, means, covariances, size=1):
      mean_vector = means[domain]  # 平均ベクトルを1次元配列として生成
      cov_matrix = covariances[domain] #分散共分散行列
      x=np.random.multivariate_normal(mean_vector, cov_matrix, size)
      return np.insert(x,0,1,axis=1)
  def __load_means_and_covariances(self):
    means={
    row['domain']: np.array([row[f'dim{i+1}'] for i in range(self.n_features)])
      for _, row in self.mean_df.iterrows() if row['data_scenario_id']==self.data_scenario_id
  } 
    covariances={}
    for _, row in self.cov_df.iterrows():
      if row['data_scenario_id']!=self.data_scenario_id:
        continue
      flat=np.array([float(x) for x in row['cov'].split(',')])
      cov=flat.reshape((self.n_features,self.n_features))
      covariances[row['domain']]=cov
    self.means=means
    self.covariances=covariances
  def __get_output(self,input,t,epsilon):
        if self.config["scenario"]["model_id"] == "linear":
            coefficient_row=self.coefficient_df[(self.coefficient_df["n_features"]==self.config["hyperparameters"]["n_features"]) & (self.coefficient_df["model_scenario_id"]==self.config["scenario"]["model_scenario_id"])]
            if coefficient_row.empty:
                raise ValueError("Coefficient not found for the specified model_id, n_features, and scenario_id.")
            coefficient = np.array([float(x) for x in coefficient_row.iloc[0]["coefficient"].split(",")])
            output = coefficient @ input.T + t + epsilon
            return np.array([output]).T
        else:
            raise ValueError(f"Unknown model_id: {self.config['scenario']['model_id']}")
  def generate_data(self,):
    self.__target_number=self.config["dataset"]["target_number"] #現在試験の被験者数
    self.__target_x=self.__generate_multivariate_normal(domain="target",means=self.means,covariances=self.covariances,size=self.__target_number) #現在試験の共変量　行が被験者、列が共変量
    self.__epsilon=np.random.normal(loc=0,scale=1,size=self.__target_number) #測定誤差
    self.__target_y=self.__get_output(
            input=self.__target_x, t=np.ones(self.__target_number), epsilon=self.__epsilon
    )

    self.__source_number = self.config["dataset"]["source_number"]
    self.__source_x=self.__generate_multivariate_normal("source",self.means,self.covariances,size=self.__source_number)
    self.__epsilon=np.random.normal(loc=0,scale=1,size=self.__source_number)
    self.__source_y=self.__get_output(
            input=self.__source_x, t=np.zeros(self.__source_number), epsilon=self.__epsilon
        )

#変数のカプセル化、意図しない値の書き換えを防ぐ目的
  @property
  def target_x(self):
     return self.__target_x
  @property
  def target_y(self):
     return self.__target_y
  @property
  def source_x(self):
    return self.__source_x
  @property
  def source_y(self):
    return self.__source_y