import pandas as pd
from workarounds.preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy

class ZScoreMethod(PreprocessingStrategy):
    def __init__(self, columns, threshold=2):
        # Accepting either single or multiple columns
        if isinstance(columns, str):
            columns = [columns]

        self.columns = columns
        self.threshold = threshold

    def fit(self, df, y=None):
        self.stats = {}
        for col in self.columns:
            self.stats[col] = {
                "mean" : df[col].mean(),
                "std" : df[col].std()
            }

        return self

    def transform(self, df):
        # Start with all rows valid
        mask = pd.Series(True, index = df.index)

        for col in self.columns:
            mean = self.stats[col]["mean"]
            std = self.stats[col]["std"]

            z = (df[col] - mean) / std
            col_mask = z.abs() < self.threshold # AND across columns

            mask &= col_mask

        return df[mask]


