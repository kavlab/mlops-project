import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Чтение данных
penguins = pd.read_csv("../data/train_penguins.csv", sep=",")

# Разделяем данные на обучающие и тестовые наборы
X = penguins.drop('species', axis=1)
y = penguins['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Оцениваем модель на тестовом множестве
score = model.score(X_test, y_test)

# Сохраняем оценку модели в файл CSV
df_score = pd.DataFrame({'evaluation_model': [score]})
df_score.to_csv("evaluation_model.csv", index=False)

# Сохранение обученной модели
pickle.dump(model, open("model_pickle.pkl", "wb"))
