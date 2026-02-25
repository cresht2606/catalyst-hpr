# 🏠 Catalyst Project - House Price Prediction

<a href="https://github.com/cresht2606/catalyst-hpr">
  <img src="https://img.shields.io/badge/Development-Stable-green?logo=github" alt="Development Stable" />
</a>
<a href="https://scikit-learn.org/">
  <img src="https://img.shields.io/badge/ML-Regression-blueviolet?logo=scikitlearn" alt="Machine Learning Regression" />
</a>
<a href="https://shields.io/">
  <img src="https://img.shields.io/badge/Stage-Release-brightgreen" alt="Release Stage" />
</a>

A dedicated and pioneering Data Science project concentrates on predicting real-time real estate based on property features, location and temporal data. The project follows rigorous successive practices, including systematic data collection, exploratory data analysis, feature engineering - preprocessing, model training and evaluation. This workflow is designed to produce experimental and interpretable models that reflect market behavior and enhance decision-making in housing markets.

## Getting Started
Before exploring the notebooks or running the models, you must setup your local environment to ensure all dependencies are met.

### Prerequisites
* **Conda** (Anaconda or Miniconda)
* **Python 3.10+**

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/cresht2606/catalyst-hpr.git](https://github.com/cresht2606/catalyst-hpr.git)
    cd catalyst-hpr
    ```

2.  **Create and activate the environment:**
    The `environment.yml` file contains all necessary libraries (pandas, scikit-learn, streamlit, etc.).
    ```bash
    conda env create -f environment.yml
    conda activate catalyst-hpr
    ```

3.  **Install the local package:**
    To ensure the `workarounds` module is accessible globally within your environment:
    ```bash
    pip install -e .
    ```

---
## Project Organization

```
├── data
│   ├── raw                 <- Original, immutable real estate listings
│   └── processed           <- Cleaned and feature-engineered datasets for ML
│
├── models                  <- Serialized preprocessing pipelines and trained models
│
├── notebooks               <- Exploratory analysis and experimental workflows
│
├── reports                 <- Streamlit-ready application for EDA and data insights
│                            (interactive dashboards and visual exploration)
│
├── tests                   <- Unit and integration tests
│
├── workarounds              <- Core project logic and utilities
│   ├── preprocessing        <- Data cleaning, imputation, encoding, and scaling
│   └── scraping             <- Scripts for collecting real estate listings
│
├── environment.yml          <- Conda environment specification
├── pyproject.toml           <- Project metadata and tool configuration
├── setup.cfg                <- Linting and formatting configuration
├── Makefile                 <- Convenience commands for common tasks
├── LICENSE                  <- Open-source license
└── README.md                <- Project overview and usage documentation
```
---
## Running the Notebooks

Once the environment is activated, you can launch Jupyter to explore the systematic data collection and model evaluation:

```bash
jupyter notebook
```
--------

