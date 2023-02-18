import pandas as pd

def fe_1(df_dict, selected_features, target_variable):
    df_dict_trans = dict()
    for key in df_dict.keys():
        df = df_dict[key].copy()
        try:
            df = df[selected_features + target_variable]
        except:
            df = df[selected_features]
        df_dict_trans[key] = df

    return df_dict_trans


def main(df_dict, exec_feature_engineering):
    func_dict = {
        "fe_1" : fe_1 
    }
    df_dict_trans = func_dict[exec_feature_engineering["name"]](df_dict=df_dict, **exec_feature_engineering["parameters"])
    return df_dict_trans