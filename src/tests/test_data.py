import pandas as pd
from collections import Counter
from scipy.stats import zscore
import numpy as np


# Функция для загрузки датасета
def load_penguins_dataset():
    return pd.read_csv('datasets/penguins.csv')


def test_missing_values_in_penguins():
    # Загрузка датасета
    df = load_penguins_dataset()

    # Проверка на наличие пропусков
    missing_values = df.isnull().sum().sum()

    assert missing_values == 0, f"В датасете обнаружено {missing_values} пропущенных значений."


def test_duplicate_values_in_penguins():
    # Загрузка датасета
    df = load_penguins_dataset()

    # Проверка на наличие дубликатов
    duplicates = set([tuple(x) for x in df.to_numpy()])
    counts = Counter(duplicates)
    dupe_counts = [(count, key) for key, count in counts.items() if count > 1]

    assert len(dupe_counts) == 0, f"В датасете обнаружены дублирующиеся записи:\n{dupe_counts}"


def test_anomalous_values_in_penguins():
    # Загрузка датасета
    df = load_penguins_dataset()

    # Преобразование типов данных
    df1 = df
    df1[['bill_length_mm', 'bill_depth_mm', 'body_mass_g']] = df1[['bill_length_mm', 'bill_depth_mm', 'body_mass_g']].astype('float64')


    # Проверка на наличие аномальных значений в столбцах 2, 3, 5 и 7
    z_scores = zscore(df1.iloc[:, [2, 3, 5]])
    abs_z_scores = np.abs(z_scores)
    threshold = 3
    anomalies = df[(abs_z_scores > threshold).any(axis=1)]

    if not anomalies.empty:
        print(f"Аномальные значения найдены в следующих строках: {anomalies.index}")
        assert False, "В датасете обнаружены аномальные значения."
    else:
        print("Все значения в пределах нормы.")
        assert True
