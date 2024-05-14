import pandas as pd

df_score = pd.read_csv("../src/models/evaluation_model.csv", sep=",")


def test_score_model():
    assert df_score['evaluation_model'][0] >= 0.90
