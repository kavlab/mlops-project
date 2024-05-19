import pandas as pd
from collections import Counter

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
