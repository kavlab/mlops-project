import pandas as pd


# Функция для загрузки датасета
def load_penguins_dataset():
    return pd.read_csv('datasets/penguins.csv')


def test_missing_values_in_penguins():
    # Загрузка датасета
    df = load_penguins_dataset()

    # Проверка на наличие пропусков
    missing_values = df.isnull().sum().sum()

    assert missing_values == 0, f"В датасете обнаружено {missing_values} пропущенных значений."
