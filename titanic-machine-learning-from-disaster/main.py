import yaml
import os
import pandas as pd
import sklearn.model_selection
import numpy as np
import datetime
import pickle

import src.feature_engineering 
import src.models

def exec_load_config():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as config_file:
        try:
            config = yaml.safe_load(config_file)
        except:
            raise("Problems loading config file...")
        config_file.close()
        return config

def exec_load_experiment(experiment_name):
    with open(os.path.join(os.path.dirname(__file__), "experiments_config/", experiment_name + ".yaml"), "r") as config_file:
        try:
            exp = yaml.safe_load(config_file)
        except:
            raise("Problems loading experiment file...")
    config_file.close()
    return exp

def exec_load_data(exec_load_data):
    df_train = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/", "train.csv"))
    df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/", "test.csv"))
    df_train_train, df_train_test = sklearn.model_selection.train_test_split(df_train, **exec_load_data["train_test_split"])
    df_dict = {
        "df_train_train" : df_train_train,
        "df_train_test" : df_train_test,
        "df_test" : df_test
    }
    return df_dict

def exec_feature_engineering(df_dict, exec_feature_engineering):
    df_dict_trans = src.feature_engineering.main(df_dict = df_dict, exec_feature_engineering = exec_feature_engineering)
    return df_dict_trans

def exec_model(df_dict_trans, exec_model):
    df_dict_trans_scored, model = src.models.main(df_dict_trans=df_dict_trans, exec_model=exec_model)
    return df_dict_trans_scored, model

def exec_performance(df_dict_trans_scored, y = "Survived", y_hat = "y_hat"):
    results = dict()
    for key in df_dict_trans_scored.keys():
        df = df_dict_trans_scored[key].copy()
        try:
            acc = np.round((np.sum(df[y] == df[y_hat])/len(df[y]))*100,3)
            results[key] = "{} accuracy : {}".format(key, acc)
        except:
            pass
    return results

def exec_save_experiment(
    save_path,
    exp,
    df_dict_trans_scored,
    model,
    results
):
    date = str(datetime.now())
    path = os.join.path(save_path, date)
    try:
        os.makedirs(path)
    except:
        pass
    # save dataframes
    for key in df_dict_trans_scored.keys():
        df_dict_trans_scored[key].to_csv(os.join.path(path, key+".csv"))
    # save model
    pickle.dump(model, open(os.join.path(path, "model.pkl"), 'wb'))
    # save results and exp
    with open(os.join.path(path, "results.yaml"), 'w') as save_file:
        yaml.dump(results, save_file)
    with open(os.join.path(path, "experiment_config.yaml"), 'w') as save_file:
        yaml.dump(exp, save_file)

if __name__ == '__main__':
    config = exec_load_config()
    exp = exec_load_experiment(config["experiment_name"])
    df_dict = exec_load_data(exp["exec_load_data"])
    df_dict_trans = exec_feature_engineering(df_dict = df_dict, exec_feature_engineering = exp["exec_feature_engineering"])
    df_dict_trans_scored, model = exec_model()
    results = exec_performance(df_dict_trans_scored=df_dict_trans_scored)
    exec_save_experiment(
        save_path = config["save_path"],
        exp=exp,
        df_dict_trans_scored = df_dict_trans_scored,
        model = model,
        results = results
        )