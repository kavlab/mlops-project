import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


def test_score_model():
    with open('trained_models/model_pickle.pkl', 'rb') as f:
        model = pickle.load(f)

        df = pd.read_csv('datasets/penguins.csv', sep=',')
        X = df.drop('species', axis=1)
        y = df['species']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)

        score = model.score(X_test, y_test)

        assert score >= 0.90
