import pandas as pd
import numpy as np
import pickle
import streamlit as st
from src.features.build_features import get_transform_column


ISLAND_LIST = ['Biscoe', 'Dream', 'Torgersen']
BILL_LENGTH_MM_MIN = 20.0
BILL_LENGTH_MM_MAX = 70.0
BILL_DEPTH_MM_MIN = 10.0
BILL_DEPTH_MM_MAX = 30.0
FLIPPER_LENGTH_MM_MIN = 160.0
FLIPPER_LENGTH_MM_MAX = 250.0
BODY_MASS_G_MIN = 2000.0
BODY_MASS_G_MAX = 8000.0
SEX_LIST = ['male', 'female']
YEAR_MIN = 2007
YEAR_MAX = 2009


def get_model_load():
    with open("trained_models/model_pickle.pkl", "rb") as f:
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
    island = st.selectbox("Остров", ISLAND_LIST)
    bill_length_mm = st.number_input("Длина клюва (мм)",
                                     min_value=BILL_LENGTH_MM_MIN,
                                     max_value=BILL_LENGTH_MM_MAX)
    bill_depth_mm = st.number_input("Глубина клюва (мм)",
                                    min_value=BILL_DEPTH_MM_MIN,
                                    max_value=BILL_DEPTH_MM_MAX)
    flipper_length_mm = st.number_input("Длина ласт (мм)",
                                        min_value=FLIPPER_LENGTH_MM_MIN,
                                        max_value=FLIPPER_LENGTH_MM_MAX)
    body_mass_g = st.number_input("Масса тела (г)",
                                  min_value=BODY_MASS_G_MIN,
                                  max_value=BODY_MASS_G_MAX)
    sex = st.selectbox("Пол", SEX_LIST)
    year = st.number_input("Год",
                           min_value=YEAR_MIN,
                           max_value=YEAR_MAX)
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
    with open("trained_models/columns.pkl", "rb") as f:
        columns = pickle.load(f)
        update_penguins = update_penguins[columns]

    # Делаем предсказания для новых данных
    y_pred = model.predict(update_penguins)
    st.write("Предполагаемый вид пингвина:", y_pred[0])
