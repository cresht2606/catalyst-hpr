from sklearn.preprocessing import RobustScaler
from workarounds.preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy

class RobustScaling(PreprocessingStrategy):
    def __init__(self, columns):
        self.columns = columns
        self.scaler = RobustScaler()

    def fit(self, df, y=None):
        self.scaler.fit(df[self.columns])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df