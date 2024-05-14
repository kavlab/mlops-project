import pandas as pd

penguins = pd.read_csv("../data/penguins.csv", sep=",")


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


# Обновленный датафрейм
update_penguins = get_transform_column(penguins)

# Сохраняем наборы данных в файл CSV
update_penguins.to_csv("../data/train_penguins.csv", index=False)
