import os
from palmerpenguins import load_penguins

# Загружаем данные
penguins = load_penguins()

# Текущая рабочая папка
current_dir = os.getcwd()

# Переходим на один уровень вверх, чтобы выйти из папки src
os.chdir("..")

# Создание папки datasets, если она не существует
os.makedirs('datasets', exist_ok=True)

# Сохраняем наборы данных в файл CSV
penguins.to_csv("datasets/penguins.csv", index=False)
