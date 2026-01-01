from sklearn.preprocessing import PowerTransformer
from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy

class BoxCoxTransformer(PreprocessingStrategy):
    def __init__(self, column):
        self.column = column
        self.pt = PowerTransformer(method="box-cox")

    def fit(self, df, y=None):
        # Fit PowerTransformer on the column (requires all positive values)
        self.pt.fit(df[[self.column]])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.column] = self.pt.transform(df[[self.column]])
        return df
    
