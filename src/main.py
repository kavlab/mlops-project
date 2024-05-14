import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from palmerpenguins import load_penguins
from sklearn.ensemble import RandomForestClassifier

# Загружаем данные
penguins = load_penguins()

# Текущая рабочая папка
current_dir = os.getcwd()

# Переходим на один уровень вверх, чтобы выйти из папки src
os.chdir("..")

# Создание папки data, если она не существует
os.makedirs('data', exist_ok=True)

# Сохраняем наборы данных в файл CSV
penguins.to_csv("data/penguins.csv", index=False)

# Возвращаемся в исходную рабочую папку
os.chdir(current_dir)

# Преобразуем категориальные столбцы в числовые
penguins['island'] = pd.Categorical(penguins['island'])
penguins['sex'] = pd.Categorical(penguins['sex'])
penguins = pd.get_dummies(penguins, columns=['island', 'sex'])

# Обработка пропущенных значений
penguins.dropna(inplace=True)

# Разделяем данные на обучающие и тестовые наборы
X = penguins.drop('species', axis=1)
y = penguins['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Оцениваем модель на тестовом множестве
score = model.score(X_test, y_test)
print('Оценка модели:', score)
print ()

#примеры для заполнения
print('Далее Вам необходимо ввести данные для предсказания вида пингвина, можете воспользоваться подготовленными примерами')
print('Пример 1. ожидаемый ответ - Gentoo,      данные: Biscoe      46.1, 15.1, 215.0, 5100.0, male, 2007')
print('Пример 2. ожидаемый ответ - Chinstrap,   данные: Dream       49.5, 19.0, 200.0, 3800.0, male, 2008')
print('Пример 3. ожидаемый ответ - Adelie,      данные: Torgersen   34.1, 18.1, 193.0, 3475.0, male, 2007')
print()

# Вводим данные о характеристиках пингвинов
while True:
    try:
        island = input("Введите остров (Biscoe, Dream, Torgersen): ")
        bill_length_mm = float(input("Введите длину клюва (мм): "))
        bill_depth_mm = float(input("Введите глубину клюва (мм): "))
        flipper_length_mm = float(input("Введите длину ласт (мм): "))
        body_mass_g = float(input("Введите массу тела (г): "))
        sex = input("Введите пол (male, female): ")
        year = input("Введите год: ")
    except ValueError:
        print("Неверно введены данные. Попробуйте снова.")
    else:
        break

new_penguins = pd.DataFrame({
    'island': [island],
    'bill_length_mm': [bill_length_mm],
    'bill_depth_mm': [bill_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex],
    'year': [year]
})

# Преобразуем категориальные столбцы в числовые
new_penguins['island'] = pd.Categorical(new_penguins['island'])
new_penguins['sex'] = pd.Categorical(new_penguins['sex'])
new_penguins = pd.get_dummies(new_penguins, columns=['island', 'sex'])

# Создаем фиктивные столбцы для столбцов, отсутствующих в обученной модели
new_penguins['island_Dream'] = np.zeros(new_penguins.shape[0])
new_penguins['island_Biscoe'] = np.zeros(new_penguins.shape[0])
new_penguins['island_Torgersen'] = np.zeros(new_penguins.shape[0])
new_penguins['sex_male'] = np.zeros(new_penguins.shape[0])
new_penguins['sex_female'] = np.zeros(new_penguins.shape[0])

# Обработка пропущенных значений
new_penguins.dropna(inplace=True)

# **Сортируем данные по столбцам так же, как в обучающих данных**
new_penguins = new_penguins[X_train.columns]

# Делаем предсказания для новых данных
y_pred = model.predict(new_penguins)

# Выводим предсказания
print('Предполагаемый вид пингвина: ', y_pred)

