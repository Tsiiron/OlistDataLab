import os
import pandas as pd

def load_data(data_dir="dataset"):
    """
    Загружает все CSV-файлы из папки dataset в словарь dfs.
    """
    dfs = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            path = os.path.join(data_dir, filename)
            df_name = filename.replace(".csv", "")
            dfs[df_name] = pd.read_csv(path)
            print(f"[LOAD] {df_name} ({dfs[df_name].shape[0]} строк)")
    return dfs

