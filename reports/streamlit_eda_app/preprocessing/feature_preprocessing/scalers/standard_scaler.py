from sklearn.preprocessing import StandardScaler
from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy

class StandardScaling(PreprocessingStrategy):
    def __init__(self, columns):
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, df, y=None):
        self.scaler.fit(df[self.columns])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df