import pandas as pd
from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy

class IQRMethod(PreprocessingStrategy):
    def __init__(self, columns):
        # Accepting either single or multiple columns
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns

    def fit(self, df, y=None):
        self.stats = {}

        for col in self.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            self.stats[col] = {"q1" : q1, "q3" : q3, "iqr" : iqr}

        return self

    def transform(self, df):
        mask = pd.Series(True, index = df.index)

        # Combine constraints from all columns
        for col in self.columns:
            q1 = self.stats[col]["q1"]
            q3 = self.stats[col]["q3"]
            iqr = self.stats[col]["iqr"]

            col_mask = (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)
            mask &= col_mask

        return df[mask]
