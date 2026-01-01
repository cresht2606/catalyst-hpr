from sklearn.preprocessing import PowerTransformer
from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy

class YeoJohnsonTransformer(PreprocessingStrategy):
    def __init__(self, column):
        self.column = column
        self.pt = PowerTransformer(method="yeo-johnson")

    def fit(self, df, y=None):
        # Fit PowerTransformer on the column (supports zero/negative values)
        self.pt.fit(df[[self.column]])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.column] = self.pt.transform(df[[self.column]])
        return df
    