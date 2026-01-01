import pandas as pd
import os

DATA_DIR = os.path.join("data", "raw")

def load_buying_properties(filename = "house_buying_sep21st_2025.csv"):
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)

def load_rental_properties(filename = "house_rental_sep21st_2025.csv"):
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)