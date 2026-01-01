from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy

class PreprocessingPipeline:
    def __init__(self, steps : list[PreprocessingStrategy]):
        self.steps = steps

    def fit(self, df, y = None):
        for step in self.steps:
            step.fit(df, y)
            df = step.transform(df)
        return self

    def transform(self, df):
        for step in self.steps:
            df = step.transform(df)
        return df
    
    def fit_transform(self, df, y = None):
        self.fit(df, y)
        return self.transform(df)