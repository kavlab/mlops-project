FROM python:3.9

#Установка зависимостей
COPY src/ui/requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

#Копирование файлов проекта
COPY src/features/build_features.py /app/
COPY src/ui/*.py /app/
COPY src/trained_models/* /app/trained_models/

WORKDIR /app

# Запуск Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
