import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Чтение данных
penguins = pd.read_csv("datasets/penguins.csv", sep=",")

# Разделяем данные на обучающие и тестовые наборы
X = penguins.drop('species', axis=1)
y = penguins['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Оцениваем модель на тестовом множестве
score = model.score(X_test, y_test)
print(f"\nModel score: {score}\n")

# Сохранение обученной модели
pickle.dump(model, open("trained_models/model_pickle.pkl", "wb"))
pickle.dump(X_train.columns, open("trained_models/columns.pkl", "wb"))
