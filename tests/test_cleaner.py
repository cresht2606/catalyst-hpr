import sys
import pandas as pd
from pathlib import Path

# Add project folder to recognize workarounds filepath
project_root = Path(__file__).resolve().parents[1] # catalyst/
sys.path.append(str(project_root))

from workarounds.preprocessing.data_cleaning.cleaner import DataCleaner

def main():
    # Sample data
    df = pd.DataFrame({
        "date": ["2025-10-01", "2025-10-02", "invalid"],
        "price": [100, None, 200],
        "desc": ["  apple ", "banana ", "apple, orange"]
    })

    # Define export folder (real path)
    export_dir = Path("data/interim")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Run full cleaning pipeline
    cleaner = (
        DataCleaner(df, interim_dir=str(export_dir))
        .strip_spaces()
        .handle_nan("mean")
        .format_dates("date")
        .split_composite_columns("desc", ["fruit1", "fruit2"], sep=",")
        .drop_duplicates()
    )

    # Export to CSV
    csv_path = cleaner.export("cleaned_data", format="csv")
    print(f"CSV exported to: {csv_path}")

    # Export to Excel
    excel_path = cleaner.export("cleaned_data", format="xlsx")
    print(f"Excel exported to: {excel_path}")

    # Get cleaned DataFrame
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned DataFrame:")
    print(cleaned_df)


if __name__ == "__main__":
    main()
