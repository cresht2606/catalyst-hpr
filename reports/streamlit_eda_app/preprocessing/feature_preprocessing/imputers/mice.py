from preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge

class MICEImputation(PreprocessingStrategy):
    def __init__(self, columns, **imputer_kwargs):
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.imputer = IterativeImputer(
            estimator=BayesianRidge(),
            **imputer_kwargs
        )

    def fit(self, df, y=None):
        self.imputer.fit(df[self.columns])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.columns] = self.imputer.transform(df[self.columns])
        return df
