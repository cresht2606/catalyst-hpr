import sys
from pathlib import Path

# -----------------------------
# Project import setup
# -----------------------------
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
from workarounds.preprocessing.feature_preprocessing.encoders.label_encoder import LabelEncoding
from workarounds.preprocessing.feature_preprocessing.encoders.target_encoder import TargetEncoding
from workarounds.preprocessing.feature_preprocessing.encoders.one_hot_encoder import OneHotEncoding
from workarounds.preprocessing.feature_preprocessing.scalers.minmax_scaler import MinMaxScaling
from workarounds.preprocessing.feature_preprocessing.scalers.standard_scaler import StandardScaling
from workarounds.preprocessing.feature_preprocessing.normalization.normalizer import Normalizing
from workarounds.preprocessing.feature_preprocessing.outliers.iqr_removal import IQRMethod
from workarounds.preprocessing.feature_preprocessing.outliers.zscore_removal import ZScoreMethod
from workarounds.preprocessing.feature_preprocessing.transformers.boxcox import BoxCoxTransformer
from workarounds.preprocessing.feature_preprocessing.transformers.yeo_johnson import YeoJohnsonTransformer
from workarounds.preprocessing.feature_preprocessing.pipeline import PreprocessingPipeline  # ← your pipeline class

# -----------------------------
# Sample dataset
# -----------------------------
df = pd.DataFrame({
    "color": ["red", "blue", "green", "red", "green"],
    "city": ["NY", "LA", "NY", "SF", "LA"],
    "category": ["A", "B", "A", "C", "B"],
    "target": [10, 20, 15, 5, 25],
    "x1": [1, 2, 3, 4, 5],
    "x2": [100, 200, 300, 400, 500],
    "feature": [1, 2, 3, 4, 5]
})

print("Original DataFrame:")
print(df)
print("\n")

# -----------------------------
# Define preprocessing pipeline
# -----------------------------
pipeline = PreprocessingPipeline([
    IQRMethod(column="x2"),                     # Remove outliers
    LabelEncoding(column="color"),              # Encode categorical
    OneHotEncoding(column="city"),              # One-hot encode
    TargetEncoding(column="category", target="target"),  # Target encode
    YeoJohnsonTransformer(column="feature"),      # Normalize skewed data
    StandardScaling(columns=["x1", "x2"]),      # Standard scale numeric
])

# -----------------------------
# Execute preprocessing
# -----------------------------
processed_df = pipeline.fit_transform(df)

print("Processed DataFrame:")
print(processed_df)
