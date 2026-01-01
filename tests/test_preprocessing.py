import pytest, sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project folder to recognize workarounds filepath
project_root = Path(__file__).resolve().parents[1] # catalyst/
sys.path.append(str(project_root))

# ============================================================
#                      IMPORTS (Adjusted)
# ============================================================

from workarounds.preprocessing.feature_preprocessing.encoders.label_encoder import LabelEncoding
from workarounds.preprocessing.feature_preprocessing.encoders.target_encoder import TargetEncoding
from workarounds.preprocessing.feature_preprocessing.encoders.one_hot_encoder import OneHotEncoding
from workarounds.preprocessing.feature_preprocessing.scalers.minmax_scaler import MinMaxScaling
from workarounds.preprocessing.feature_preprocessing.scalers.standard_scaler import StandardScaling
from workarounds.preprocessing.feature_preprocessing.normalization.normalizer import Normalizing
from workarounds.preprocessing.feature_preprocessing.outliers.iqr_removal import IQRMethod
from workarounds.preprocessing.feature_preprocessing.outliers.zscore_removal import ZScoreMethod
from workarounds.preprocessing.feature_preprocessing.transformers.boxcox import BoxCoxTransform
from workarounds.preprocessing.feature_preprocessing.transformers.yeo_johnson import YeoJohnsonTransform
from workarounds.preprocessing.feature_preprocessing.context import PreprocessorContext


# ============================================================
#                      PYTEST FIXTURES
# ============================================================

@pytest.fixture
def categorical_df():
    """Simple categorical DataFrame."""
    return pd.DataFrame({"color": ["red", "blue", "green", "red"]})


@pytest.fixture
def city_df():
    """Simple DataFrame for one-hot encoding."""
    return pd.DataFrame({"city": ["NY", "LA", "NY", "SF"]})


@pytest.fixture
def target_df():
    """DataFrame for target encoding."""
    return pd.DataFrame({
        "category": ["A", "B", "A", "C"],
        "target": [10, 20, 15, 5]
    })


@pytest.fixture
def numeric_df():
    """DataFrame for scaling and normalization."""
    return pd.DataFrame({"x1": [1, 2, 3, 4]})


@pytest.fixture
def wide_range_df():
    """DataFrame for standard scaling test."""
    return pd.DataFrame({"x1": [10, 20, 30, 40]})


@pytest.fixture
def outlier_df():
    """DataFrame containing outliers for IQR/Z-score tests."""
    return pd.DataFrame({"value": [10, 12, 14, 1000]})


@pytest.fixture
def boxcox_df():
    """Positive-only data for Box-Cox transform."""
    return pd.DataFrame({"feature": [1, 2, 3, 4]})


@pytest.fixture
def yeojohnson_df():
    """Data with negatives and zeros for Yeo-Johnson transform."""
    return pd.DataFrame({"feature": [-2, -1, 0, 1, 2]})


# ============================================================
#                      TEST CASES
# ============================================================

def test_label_encoding(categorical_df):
    strategy = LabelEncoding(column="color")
    context = PreprocessorContext(strategy)
    df_processed = context.execute(categorical_df.copy())

    assert np.issubdtype(df_processed["color"].dtype, np.integer)
    assert set(df_processed["color"].unique()) == {0, 1, 2}


def test_one_hot_encoding(city_df):
    strategy = OneHotEncoding(column="city")
    context = PreprocessorContext(strategy)
    df_processed = context.execute(city_df.copy())

    assert "city_NY" in df_processed.columns
    assert "city_LA" in df_processed.columns
    assert df_processed.shape[1] > city_df.shape[1]


def test_target_encoding(target_df):
    strategy = TargetEncoding(column="category", target="target")
    context = PreprocessorContext(strategy)
    df_processed = context.execute(target_df.copy())

    assert np.issubdtype(df_processed["category"].dtype, np.number)
    assert not df_processed["category"].isnull().any()


def test_minmax_scaling(numeric_df):
    """Tests the MinMaxScaler wrapper."""
    strategy = MinMaxScaling(columns=["x1"])
    context = PreprocessorContext(strategy)
    df_scaled = context.execute(numeric_df.copy())

    assert df_scaled["x1"].min() == pytest.approx(0.0)
    assert df_scaled["x1"].max() == pytest.approx(1.0)


def test_standard_scaling(wide_range_df):
    """Tests the StandardScaler wrapper."""
    strategy = StandardScaling(columns=["x1"])
    context = PreprocessorContext(strategy)
    df_scaled = context.execute(wide_range_df.copy())

    mean_approx = np.round(df_scaled["x1"].mean(), 6)
    std_approx = np.round(df_scaled["x1"].std(), 6)

    assert np.isclose(mean_approx, 0, atol=1e-2)
    assert np.isclose(std_approx, 1, atol=1e-2)


def test_normalizer(numeric_df):
    """Tests normalization using L2 norm."""
    strategy = Normalizing(columns=["x1"])
    context = PreprocessorContext(strategy)
    df_norm = context.execute(numeric_df.copy())

    norm = np.linalg.norm(df_norm["x1"])
    assert norm == pytest.approx(1.0)


def test_iqr_method_removes_outliers(outlier_df):
    strategy = IQRMethod(column="value")
    context = PreprocessorContext(strategy)
    df_filtered = context.execute(outlier_df.copy())

    assert 1000 not in df_filtered["value"].values
    assert len(df_filtered) < len(outlier_df)


def test_zscore_method_removes_outliers(outlier_df):
    strategy = ZScoreMethod(column="value", threshold=2)
    context = PreprocessorContext(strategy)
    df_filtered = context.execute(outlier_df.copy())

    assert 1000 not in df_filtered["value"].values
    assert len(df_filtered) < len(outlier_df)


def test_boxcox_transform(boxcox_df):
    strategy = BoxCoxTransform(column="feature")
    context = PreprocessorContext(strategy)
    df_transformed = context.execute(boxcox_df.copy())

    assert not df_transformed["feature"].isnull().any()
    assert np.isfinite(df_transformed["feature"]).all()


def test_yeojohnson_transform(yeojohnson_df):
    strategy = YeoJohnsonTransform(column="feature")
    context = PreprocessorContext(strategy)
    df_transformed = context.execute(yeojohnson_df.copy())

    assert not df_transformed["feature"].isnull().any()
    assert np.isfinite(df_transformed["feature"]).all()
