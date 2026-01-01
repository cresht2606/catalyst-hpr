from workarounds.preprocessing.feature_preprocessing.strategy_base import PreprocessingStrategy
import pandas as pd

class OneHotEncoding(PreprocessingStrategy):
    def __init__(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.feature_names_ = []
        self.categories_ = {}

    def fit(self, df, y=None):
        df = df.copy()
        self.feature_names_ = []
        self.categories_ = {}

        for col in self.columns:
            cats = df[col].astype(str).unique().tolist()
            self.categories_[col] = cats
            self.feature_names_ += [f"{col}_{cat}" for cat in cats]

        return self

    def transform(self, df):
        df = df.copy()

        ohe_data = {}

        for col in self.columns:
            col_values = df[col].astype(str)
            for cat in self.categories_[col]:
                ohe_data[f"{col}_{cat}"] = (col_values == cat).astype(int)

        # Create OHE dataframe in one go
        ohe_df = pd.DataFrame(ohe_data, index = df.index)

        # Ensure consistent column order
        ohe_df = ohe_df.reindex(columns=self.feature_names_, fill_value=0)

        # Drop original categorical columns
        df = df.drop(columns = self.columns)

        # Concatenate once
        df = pd.concat([df, ohe_df], axis = 1)

        return df
