def md_1(args):
    pass

def main(df_dict_trans, exec_model):
    models_dict = {
        "md_1" : md_1
    }
    df_dict_trans_scored, model = models_dict[exec_model["name"]](df_dict_trans=df_dict_trans, **exec_model["parameters"])