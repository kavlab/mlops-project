import streamlit as st
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

# Добавляем примеры
st.title('Примеры для заполнения')
st.write('Пример 1. ожидаемый ответ - Gentoo,      данные: Biscoe      46.1, 15.1, 215.0, 5100.0, male, 2007')
st.write('Пример 2. ожидаемый ответ - Chinstrap,   данные: Dream       49.5, 19.0, 200.0, 3800.0, male, 2008')
st.write('Пример 3. ожидаемый ответ - Adelie,      данные: Torgersen   34.1, 18.1, 193.0, 3475.0, male, 2007')

# Создаем приложение Streamlit
st.title("Предиктор вида пингвина")

# Создаем форму для ввода данных
with st.form("form"):
    island = st.selectbox("Остров", ['Biscoe', 'Dream', 'Torgersen'])
    bill_length_mm = st.number_input("Длина клюва (мм)", min_value=0.0)
    bill_depth_mm = st.number_input("Глубина клюва (мм)", min_value=0.0)
    flipper_length_mm = st.number_input("Длина ласт (мм)", min_value=0.0)
    body_mass_g = st.number_input("Масса тела (г)", min_value=0.0)
    sex = st.selectbox("Пол", ['male', 'female'])
    year = st.number_input("Год", min_value=2007, max_value=2009)
    submitted = st.form_submit_button("Узнать вид пингвина")

# Делаем предсказание, если пользователь отправил форму
if submitted:
    new_penguins = pd.DataFrame({
        'island': [island],
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex],
        'year': [year]
    })
    new_penguins['island'] = pd.Categorical(new_penguins['island'])
    new_penguins['sex'] = pd.Categorical(new_penguins['sex'])
    new_penguins = pd.get_dummies(new_penguins, columns=['island', 'sex'])
    new_penguins['island_Dream'] = np.zeros(new_penguins.shape[0])
    new_penguins['island_Biscoe'] = np.zeros(new_penguins.shape[0])
    new_penguins['island_Torgersen'] = np.zeros(new_penguins.shape[0])
    new_penguins['sex_male'] = np.zeros(new_penguins.shape[0])
    new_penguins['sex_female'] = np.zeros(new_penguins.shape[0])
    new_penguins.dropna(inplace=True)
    new_penguins = new_penguins[X_train.columns]
    y_pred = model.predict(new_penguins)
    st.write("Предполагаемый вид пингвина:", y_pred[0])
