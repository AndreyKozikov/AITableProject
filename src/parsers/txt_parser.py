import pandas as pd
from src.utils.df_utils import write_to_json


def load_txt_file(path):
    df = pd.read_csv(path).to_list()
    print(df)
    #write_to_json(df, path)
    #return df