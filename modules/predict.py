# <YOUR_IMPORTS>
import datetime
import glob
import json
import os

import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH')
local = 'D:/data_science/модуль-33/airflow_hw'


def load_model():
    with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)
    return model


def predict():
    # Укажите путь к директории с JSON-файлами
    json_dir = f'{path}/data/test'
    json_pattern = os.path.join(json_dir, '*.json')
    file_list = glob.glob(json_pattern)

    model = load_model()

    dfs = []  # Список для хранения датафреймов

    for file in file_list:
        with open(file) as f:
            json_data = pd.json_normalize(json.loads(f.read()))  # Преобразование JSON в датафрейм
            dfs.append(json_data)  # Добавление датафрейма в список

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['pred'] = model.predict(combined_df)
    combined_df[['id', 'pred']].to_csv(
        f'{path}/data/predictions/predictions_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
