import pandas as pd


def get_transform_column(df):
    """
    Преобразует категориальные столбцы в числовые. Обрабатывает пропущенные значения.
    :param df:
    :return df:
    """

    df['island'] = pd.Categorical(df['island'])
    df['sex'] = pd.Categorical(df['sex'])
    df = pd.get_dummies(df, columns=['island', 'sex'])

    df.dropna(inplace=True)

    return df


penguins = pd.read_csv("datasets/penguins.csv", sep=",")

# Обновленный датафрейм
update_penguins = get_transform_column(penguins)

# Сохраняем наборы данных в файл CSV
update_penguins.to_csv("datasets/penguins.csv", index=False)
