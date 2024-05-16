FROM python:3.9

#Установка зависимостей
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

#Копирование файлов проекта
COPY . /app
WORKDIR /app

# Обучение модели
RUN python /app/src/train_model.py

# Запуск Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
