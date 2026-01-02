from sklearn.preprocessing import MinMaxScaler
from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy

class MinMaxScaling(PreprocessingStrategy):
    def __init__(self, columns):
        self.columns = columns
        self.scaler = MinMaxScaler()

    def fit(self, df, y=None):
        self.scaler.fit(df[self.columns])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df