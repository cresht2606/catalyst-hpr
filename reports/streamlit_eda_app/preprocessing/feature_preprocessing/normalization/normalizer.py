from sklearn.preprocessing import Normalizer
from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy

class Normalizing(PreprocessingStrategy):
    def __init__(self, columns):
        self.columns = columns
        self.norm = Normalizer()
    
    def fit(self, df, y=None):
        self.norm.fit(df[self.columns])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.columns] = self.norm.transform(df[self.columns])
        return df