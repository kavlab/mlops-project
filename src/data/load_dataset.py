from palmerpenguins import load_penguins

# Загружаем данные
penguins = load_penguins()

# Сохраняем наборы данных в файл CSV
penguins.to_csv("../datasets/penguins.csv", index=False)
