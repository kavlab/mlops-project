import pandas as pd
import numpy as np
import pickle
import streamlit as st
from src.models.preprocessing import get_transform_column
from src.models.preparation import X_train


def get_model_load():
    with open("src/models/model_pickle.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = get_model_load()

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

    # Преобразуем категориальные столбцы в числовые. Обрабатываем пропущенные значения.
    update_penguins = get_transform_column(new_penguins)

    # Создаем фиктивные столбцы для столбцов, отсутствующих в обученной модели
    update_penguins['island_Dream'] = np.zeros(update_penguins.shape[0])
    update_penguins['island_Biscoe'] = np.zeros(update_penguins.shape[0])
    update_penguins['island_Torgersen'] = np.zeros(update_penguins.shape[0])
    update_penguins['sex_male'] = np.zeros(update_penguins.shape[0])
    update_penguins['sex_female'] = np.zeros(update_penguins.shape[0])

    # Сортируем данные по столбцам так же, как в обучающих данных
    update_penguins = update_penguins[X_train.columns]

    # Делаем предсказания для новых данных
    y_pred = model.predict(update_penguins)
    st.write("Предполагаемый вид пингвина:", y_pred[0])
