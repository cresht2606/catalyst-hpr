from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy
import category_encoders as ce

class TargetEncoding(PreprocessingStrategy):
    def __init__(self, columns, target):
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.target = target
        self.encoders = {col : ce.TargetEncoder() for col in self.columns}

    def fit(self, df, y=None):
        for col in self.columns:
            self.encoders[col].fit(df[col], df[self.target])
        return self

    def transform(self, df):
        df = df.copy()
        for col in self.columns:
            df[col] = self.encoders[col].transform(df[col])
        return df