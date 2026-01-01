import pandas as pd
import os
from pathlib import Path

class DataCleaner:
    def __init__(self, df: pd.DataFrame, interim_dir: str = None):
        self.df = df.copy()
        # Default interim folder: project_root / data / interim
        project_root = Path(__file__).resolve().parents[3]  # -> points to project root
        self.interim_dir = Path(interim_dir) if interim_dir else project_root / "data" / "interim"
        os.makedirs(self.interim_dir, exist_ok=True)

    # Format date columns

    def format_dates(self, column, date_format="%Y-%m-%d"):
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        self.df[column] = pd.to_datetime(self.df[column], errors="coerce", format=date_format)
        return self  # enables chaining

    # Drop duplicates

    def drop_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Dropped {before - len(self.df)} duplicate rows.")
        return self

    # Handle NaN / Missing values

    def handle_nan(self, method, columns=None):
        columns = columns or self.df.columns

        valid_methods = {"zero", "mean", "median", "mode"}
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}.")

        for col in columns:
            if self.df[col].isna().any():
                if method == "zero":
                    self.df[col] = self.df[col].fillna(0)
                elif method == "mean":
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif method == "median":
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif method == "mode":
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        return self

    # Split composite columns

    def split_composite_columns(self, column, new_columns, sep=", "):
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        self.df[new_columns] = self.df[column].str.split(sep, expand=True)
        return self

    # Strip redundant spaces

    def strip_spaces(self):
        for col in self.df.select_dtypes(include=["object", "string"]).columns:
            self.df[col] = self.df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        return self

    # Export interim data

    def export(self, filename: str, format: str = "csv"):
        """Export cleaned data to interim folder in CSV or Excel format."""
        filepath = os.path.join(self.interim_dir, f"{filename}.{format}")

        if format == "csv":
            self.df.to_csv(filepath, index=False)
        elif format in {"xlsx", "xls"}:
            self.df.to_excel(filepath, index=False)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'xlsx'.")

        print(f"Data exported to {filepath}")
        #return self  # returning self allows chaining `.get_cleaned_data()`
        return str(filepath)

    # Return the cleaned data
    
    def get_cleaned_data(self):
        return self.df
