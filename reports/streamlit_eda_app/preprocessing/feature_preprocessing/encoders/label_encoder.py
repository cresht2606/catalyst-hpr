from sklearn.preprocessing import LabelEncoder
from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy

class LabelEncoding(PreprocessingStrategy):
    def __init__(self, column):
        self.column = column
        self.encoder = LabelEncoder()

    def fit(self, df, y=None):
        self.encoder.fit(df[self.column])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.column] = self.encoder.transform(df[self.column])
        return df