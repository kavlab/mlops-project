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
YEAR_MAX = 2020


def get_model_load():
    with open("trained_models/model_pickle.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def validate(island, bill_length_mm, bill_depth_mm,
            flipper_length_mm, body_mass_g, sex, year) -> bool:
    st.session_state.is_error_form = False

    if island not in ISLAND_LIST:
        st.error("Проверьте значение поля Остров")
        st.session_state.is_error_form = True

    if not BILL_LENGTH_MM_MIN <= bill_length_mm <= BILL_LENGTH_MM_MAX:
        st.error("Проверьте значение поля Длина клюва (мм)")
        st.session_state.is_error_form = True

    if not BILL_DEPTH_MM_MIN <= bill_depth_mm <= BILL_DEPTH_MM_MAX:
        st.error("Проверьте значение поля Глубина клюва (мм)")
        st.session_state.is_error_form = True

    if not FLIPPER_LENGTH_MM_MIN <= flipper_length_mm <= FLIPPER_LENGTH_MM_MAX:
        st.error("Проверьте значение поля Длина ласт (мм)")
        st.session_state.is_error_form = True

    if not BODY_MASS_G_MIN <= body_mass_g <= BODY_MASS_G_MAX:
        st.error("Проверьте значение поля Масса тела (г)")
        st.session_state.is_error_form = True

    if sex not in SEX_LIST:
        st.error("Проверьте значение поля Пол")
        st.session_state.is_error_form = True

    if not YEAR_MIN <= year <= YEAR_MAX:
        st.error("Проверьте значение поля Год")
        st.session_state.is_error_form = True

    return st.session_state.is_error_form

def predict(model, island, bill_length_mm, bill_depth_mm,
            flipper_length_mm, body_mass_g, sex, year):
    new_penguins = pd.DataFrame({
        'island': [island],
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex],
        'year': [year]})

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
    return model.predict(update_penguins)[0]


model = get_model_load()

# Добавляем примеры
st.title('Примеры для заполнения')
st.write('Пример 1. ожидаемый ответ - Gentoo,      данные: Biscoe      46.1, 15.1, 215.0, 5100.0, male, 2007')
st.write('Пример 2. ожидаемый ответ - Chinstrap,   данные: Dream       49.5, 19.0, 200.0, 3800.0, male, 2008')
st.write('Пример 3. ожидаемый ответ - Adelie,      данные: Torgersen   34.1, 18.1, 193.0, 3475.0, male, 2007')

# Создаем приложение Streamlit
st.title("Предиктор вида пингвина")

# Создаем форму для ввода данных
with st.form(key="form"):
    st.session_state.is_error_form = False
    island = st.selectbox("Остров",
                          ISLAND_LIST,
                          key="island")
    bill_length_mm = st.number_input("Длина клюва (мм)",
                                     value=BILL_LENGTH_MM_MIN,
                                     key="bill_length_mm")
    bill_depth_mm = st.number_input("Глубина клюва (мм)",
                                    value=BILL_DEPTH_MM_MIN,
                                    key="bill_depth_mm")
    flipper_length_mm = st.number_input("Длина ласт (мм)",
                                        value=FLIPPER_LENGTH_MM_MIN,
                                        key="flipper_length_mm")
    body_mass_g = st.number_input("Масса тела (г)",
                                  value=BODY_MASS_G_MIN,
                                  key="body_mass_g")
    sex = st.selectbox("Пол",
                       SEX_LIST,
                       key="sex")
    year = st.number_input("Год",
                           value=YEAR_MIN,
                           key="year")
    submitted = st.form_submit_button("Узнать вид пингвина")

# Делаем предсказание, если пользователь отправил форму
if submitted and not st.session_state.is_error_form:
    val_res_error = validate(island, bill_length_mm, bill_depth_mm,
                             flipper_length_mm, body_mass_g, sex, year)
    if val_res_error:
        st.stop()

    result = predict(model, island, bill_length_mm, bill_depth_mm,
                     flipper_length_mm, body_mass_g, sex, year)
    st.write("Предполагаемый вид пингвина:", result)
