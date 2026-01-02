from abc import ABC, abstractmethod
import pandas as pd

class PreprocessingStrategy(ABC):

    """
    Abstract base class for preprocessing strategies.
    Defines sklearn-like interface: fit / transform / fit_transform.
    """
    
    def fit(self, df : pd.DataFrame, y = None):
        """Fit any model parameters on the data (optional)."""
        return self

    @abstractmethod
    def transform(self, df : pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation to the data."""
        pass

    def fit_transform(self, df : pd.DataFrame, y = None):
        """Convenience method to fit and then transform."""
        self.fit(df, y)
        return self.transform(df)
