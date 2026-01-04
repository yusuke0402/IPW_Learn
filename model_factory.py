import yaml
import pandas as pd
import numpy as np

class ModelFactory:
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    coefficient_df=pd.read_csv("configs/coefficient.csv")
    @staticmethod
    def get_output(input,t,epsilon):
        if ModelFactory.config["scenario"]["model_id"] == "linear":
            coefficient_row=ModelFactory.coefficient_df[(ModelFactory.coefficient_df["n_features"]==ModelFactory.config["hyperparameters"]["n_features"]) & (ModelFactory.coefficient_df["model_scenario_id"]==ModelFactory.config["scenario"]["model_scenario_id"])]
            if coefficient_row.empty:
                raise ValueError("Coefficient not found for the specified model_id, n_features, and scenario_id.")
            coefficient = np.array([float(x) for x in coefficient_row.iloc[0]["coefficient"].split(",")])
            output = coefficient @ input.T + t + epsilon
            return np.array([output]).T
        else:
            raise ValueError(f"Unknown model_id: {ModelFactory.config['scenario']['model_id']}")