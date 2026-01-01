import pytest, os
import pandas as pd

from workarounds.preprocessing.data_cleaning.cleaner import DataCleaner

# Default sample code
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": ["2025-10-01", "2025-10-02", "invalid"],
        "price": [100, None, 200],
        "desc": ["  apple ", "banana ", "apple, orange"],    
    })

def test_format_dates(sample_df):
    cleaner = DataCleaner(sample_df)
    cleaner.format_dates("date")

    assert pd.api.types.is_datetime64_any_dtype(cleaner.df["date"]) # Ensure each types is 

def test_drop_duplicates(sample_df):
    cleaner = DataCleaner(sample_df)
    cleaner.drop_duplicates()

    assert len(cleaner.df) == len(sample_df) # Check if the data is already unique

def test_handle_nan(sample_df):
    cleaner = DataCleaner(sample_df)
    cleaner.handle_nan("mean", ["price"])

    assert not cleaner.df["price"].isna().any() # Chech if there exists any failed attemps

def test_split_composite_columns(sample_df):
    cleaner = DataCleaner(sample_df)
    cleaner.split_composite_columns("desc", ["fruit1", "fruit2"], sep=",")

    assert "fruit1" in cleaner.df.columns # Check "apple" seperated column
    assert "fruit2" in cleaner.df.columns # Check "orange" seperated column

def test_strip_spaces(sample_df):
    cleaner = DataCleaner(sample_df)
    cleaner.strip_spaces()

    assert all(cleaner.df["desc"].str.startswith("apple") | cleaner.df["desc"].str.startswith("banana"))

def test_export_csv(tmp_path, sample_df):
    interim_dir = tmp_path / "interim"
    cleaner = DataCleaner(sample_df, interim_dir=str(interim_dir))

    filepath = cleaner.export("test_file", format="csv")
    assert os.path.exists(filepath)

    df_out = pd.read_csv(filepath)
    assert not df_out.empty

