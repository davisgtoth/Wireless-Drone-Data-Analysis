import os
import pandas as pd

def append_data(file_path, data):
    df = pd.DataFrame([data])

    # print("\n--- MEASUREMENT RESULTS ---")
    # print(df.T)

    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

    # print(f'--> Measurements appended to {file_path}')

    return