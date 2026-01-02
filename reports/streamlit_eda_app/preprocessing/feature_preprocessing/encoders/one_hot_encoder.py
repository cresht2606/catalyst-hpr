from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy
import pandas as pd

class OneHotEncoding(PreprocessingStrategy):
    def __init__(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns

    def fit(self, df, y=None):
        # No fitting needed
        return self

    def transform(self, df):
        df = df.copy()
        return pd.get_dummies(df, columns=self.columns, drop_first=False)
    