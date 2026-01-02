import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import streamlit as st

# Task 1

# -----------------------------
# 1. Area Distribution
# -----------------------------
def plot_area_distribution(df, lower_bound=0, upper_bound=400):
    df_area = df[(df["area_m2"] > lower_bound) & (df["area_m2"] < upper_bound)]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(data=df_area, x="area_m2", bins=15, kde=True, color="forestgreen", ax=ax)
    ax.set_title(f"Distribution of Area within {upper_bound} (m2)")
    ax.set_xlabel("Area (m2)")
    ax.set_ylabel("Counts")
    st.pyplot(fig)


# -----------------------------
# 2. Bedroom Distribution
# -----------------------------
def plot_bedroom_distribution(df):
    bins = [0, 1, 2, 3, 4, 5, 10]
    labels = ["1", "2", "3", "4", "5", "6+"]
    df_bed = df[df["bedrooms"].notna()].copy()
    df_bed["bedroom_bins"] = pd.cut(df_bed["bedrooms"], bins=bins, labels=labels, right=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_bed, x="bedroom_bins", palette="Pastel1", ax=ax)
    
    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(str(count), (p.get_x() + p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    
    ax.set_title("Bedroom Distribution")
    ax.set_xlabel("No. Bedrooms")
    ax.set_ylabel("Counts")
    st.pyplot(fig)


# -----------------------------
# 3. Bathroom Distribution
# -----------------------------
def plot_bathroom_distribution(df):
    bins = [0, 1, 2, 3, 4, 5, 10]
    labels = ["1", "2", "3", "4", "5", "6+"]
    df_bath = df[df["bathrooms"].notna()].copy()
    df_bath["bathroom_bins"] = pd.cut(df_bath["bathrooms"], bins=bins, labels=labels, right=True)
    
    bath_counts = df_bath["bathroom_bins"].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=bath_counts.index, y=bath_counts.values, palette="viridis", ax=ax)
    
    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(str(count), (p.get_x() + p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    
    ax.set_title("Bathroom Distribution")
    ax.set_xlabel("No. Bathrooms")
    ax.set_ylabel("Counts")
    st.pyplot(fig)


# -----------------------------
# 4. Floor Distribution
# -----------------------------
def plot_floor_distribution(df):
    bins = [0, 1, 2, 3, 4, 5, 10]
    labels = ["1", "2", "3", "4", "5", "6+"]
    df_floor = df[df["floors"].notna()].copy()
    df_floor["floor_bins"] = pd.cut(df_floor["floors"], bins=bins, labels=labels, right=True)
    
    floor_counts = df_floor["floor_bins"].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=floor_counts.index, y=floor_counts.values, palette="magma", ax=ax)
    
    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(str(count), (p.get_x() + p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    
    ax.set_title("Floor Distribution")
    ax.set_xlabel("No. Floors")
    ax.set_ylabel("Counts")
    st.pyplot(fig)


# -----------------------------
# 5. Price Distribution by Quantile
# -----------------------------
def plot_price_quantiles(df):
    df_price = df[(df["price_million_vnd"].notna()) & (df["price_million_vnd"] >= 0)].copy()
    q1, q2, q3 = df_price["price_million_vnd"].quantile([0.25, 0.5, 0.75])
    max_price = df_price["price_million_vnd"].max()
    
    intervals = [(0, q1), (q1, q2), (q2, q3), (q3, max_price)]
    titles = ["0 - 25% Quantile", "25% - 50% Quantile", "50% - 75% Quantile", "75% - 100% Quantile"]
    
    fig, axes = plt.subplots(1, 4, figsize=(26, 8))
    
    for i, (low, high) in enumerate(intervals):
        subset = df_price[(df_price["price_million_vnd"] > low) & (df_price["price_million_vnd"] <= high)]
        sns.histplot(subset["price_million_vnd"], bins=15, color="skyblue", kde=False, ax=axes[i])
        
        for p in axes[i].patches:
            count = int(p.get_height())
            axes[i].annotate(str(count), (p.get_x() + p.get_width()/2, p.get_height()),
                             ha="center", va="bottom", fontsize=9)
        
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Price (million VND)")
        axes[i].set_ylabel("Count")
    
    st.pyplot(fig)


# -----------------------------
# 6. Correlation Heatmap
# -----------------------------
def plot_correlation_matrix(df):
    numeric = df.select_dtypes(include=["number"]).drop(columns=["id", "timeline_hours", "year", "day", "month"], errors='ignore')
    corr_matrix = numeric.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlation Matrix of Independent & Dependent Features")
    st.pyplot(fig)


# -----------------------------
# 7. Outlier Check
# -----------------------------
def plot_outliers(df, columns, sample_frac=0.05):
    sample_df = df.sample(frac=sample_frac, random_state=42)
    
    fig, ax = plt.subplots(1, len(columns), figsize=(6*len(columns), 5))
    
    if len(columns) == 1:
        ax = [ax]
    
    for i, col in enumerate(columns):
        sns.boxplot(y=sample_df[col], ax=ax[i])
        ax[i].set_title(f"{col} - Outlier Check")
    
    st.pyplot(fig)



# Exploratory Data Analysis (Discussion)

# Task 1: Lunar Shipwreck Of Predestined Whereabouts

# -----------------------------
# 1. Location-based Insights
# -----------------------------

def plot_province_distribution_t1(hb_location):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(
        data=hb_location,
        y="province",
        order=hb_location["province"].value_counts().index,
        palette="deep",
        ax=ax
    )
    ax.set_title("Distribution Of Listings By Selective Provinces")
    ax.set_xlabel("Count")
    ax.set_ylabel("Province")

    # Add annotations
    for p in ax.patches:
        count = int(p.get_width())
        y = p.get_y() + p.get_height() / 2
        ax.text(p.get_width() + 0.5, y, count, va="center")

    st.pyplot(fig)


def plot_top_wards_t1(hb_location, top_n=20):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(
        data=hb_location,
        y="ward",
        order=hb_location["ward"].value_counts().head(top_n).index,
        palette="deep",
        ax=ax
    )
    ax.set_title(f"Top {top_n} Wards by Listing Count")
    ax.set_xlabel("Count")
    ax.set_ylabel("Ward")

    # Add annotations
    for p in ax.patches:
        count = int(p.get_width())
        y = p.get_y() + p.get_height() / 2
        ax.text(p.get_width() + 0.5, y, count, va="center")

    st.pyplot(fig)


def plot_avg_price_by_province_t1(hb_location):
    location_summary = hb_location.groupby("province").agg(
        avg_price=("price_million_vnd", "mean"),
        avg_area=("area_m2", "mean"),
        count=("id", "count")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=location_summary, x="avg_price", y="province", palette="viridis", ax=ax)

    for p in ax.patches:
        count = int(p.get_width())
        y = p.get_y() + p.get_height() / 2
        ax.text(p.get_width() + 0.5, y, count, va="center")

    ax.set_title("Average House Price by Province")
    ax.set_xlabel("Average Price (Million VND)")
    ax.set_ylabel("Province")
    st.pyplot(fig)


def plot_min_max_price_by_province_t1(hb_location):
    avg_price_province = hb_location.groupby("province")["price_million_vnd"].agg(
        ["min", "max", "mean", "count"]
    ).reset_index()

    fig, ax = plt.subplots(2, 1, figsize=(13, 12), sharex=True)

    sns.pointplot(data=avg_price_province, x="province", y="max", color="red", ax=ax[0])
    ax[0].set_title("Maximum Price by Province")
    for i, value in enumerate(avg_price_province["max"]):
        ax[0].text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=10, color="red")

    sns.pointplot(data=avg_price_province, x="province", y="min", color="blue", ax=ax[1])
    ax[1].set_title("Minimum Price by Province")
    for i, value in enumerate(avg_price_province["min"]):
        ax[1].text(i, value, f"{value:,.3f}", ha="center", va="bottom", fontsize=10, color="blue")

    plt.xticks(rotation=45)
    st.pyplot(fig)


# -----------------------------
# 2. Price Analysis
# -----------------------------

def plot_price_distribution_t1(filtered, hb_location):
    # Overall violin plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=filtered, y="price_million_vnd", ax=ax)
    ax.set_title("Distribution of House Prices - Outliers Excluded")
    ax.set_ylabel("Price (Million VND)")
    st.pyplot(fig)

    # By province
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=filtered.merge(hb_location[["id", "province"]], on="id"),
        x="province",
        y="price_million_vnd",
        palette="Accent",
        ax=ax
    )
    ax.set_title("Price Distribution by Province")
    ax.set_ylabel("Price Million VND")
    ax.set_xlabel("Province")
    plt.xticks(rotation=30)
    st.pyplot(fig)


def plot_price_per_m2_t1(filtered, hb_location):
    filtered["price_per_m2"] = filtered["price_million_vnd"] / filtered["area_m2"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=filtered.merge(hb_location[["id", "province"]], on="id"),
        x="province",
        y="price_per_m2",
        ax=ax
    )
    ax.set_xlabel("Province")
    ax.set_ylabel("Price per m2 (Million VND)")
    ax.set_title("Price per m2 By Province")
    plt.xticks(rotation=45)
    st.pyplot(fig)


def plot_price_segment_t1(filtered):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.countplot(data=filtered, x="price_segment", ax=axes[0], palette="Set3")
    axes[0].set_xlabel("Price Segmentation")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Price Segment Distribution")
    axes[0].tick_params(axis='x')

    filtered["price_segment"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=axes[1])
    axes[1].set_title("Price Segment Proportion")
    axes[1].set_ylabel("")

    st.pyplot(fig)


def plot_frontage_influence_t1(filtered):
    filtered["frontage_tag"] = filtered["frontage"].astype(bool).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=filtered, x="frontage_tag", y="price_million_vnd", showfliers=False, ax=ax)
    sns.stripplot(data=filtered, x="frontage_tag", y="price_million_vnd", color="black", alpha=0.4, jitter=True, ax=ax)

    plt.xticks([0, 1], ["Without Frontage", "With Frontage"])
    ax.set_title("Price Comparison: With VS. Without Frontage Tag")
    st.pyplot(fig)


# -----------------------------
# Price VS Floors
# -----------------------------
def plot_price_vs_floors_t1(floor_filtered):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.regplot(data=floor_filtered, x="floors", y="price_million_vnd", ax=ax)
    ax.set_xlabel("No. Floors")
    ax.set_ylabel("Price (Million VND)")
    ax.set_title("Price VS. Floors")
    st.pyplot(fig)


# -----------------------------
# Combined Factors: Frontage, Floors, Province
# -----------------------------
def plot_combined_factors_t1(filtered, hb_location):
    temp = filtered.merge(hb_location[["id", "province"]], on="id")
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.violinplot(data=temp, x="province", y="price_million_vnd", hue="frontage_tag", split=True, ax=ax)
    ax.set_xlabel("Province")
    ax.set_ylabel("Price (Million VND)")
    ax.set_title("Combined Factors Onto Price Distribution: Frontage - Floors - Province")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# 3. House Size & Layout Interpretation

# -----------------------------
# House Size vs Bedrooms/Bathrooms
# -----------------------------
def plot_area_vs_bed_bath_t1(hb_size_filtered):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Area VS Bedrooms
    sns.scatterplot(data=hb_size_filtered, x="area_m2", y="bedrooms", alpha=0.7, ax=axes[0])
    sns.regplot(data=hb_size_filtered, x="area_m2", y="bedrooms", scatter=False, ax=axes[0], color="black")
    axes[0].set_xlabel("Area (m2)")
    axes[0].set_ylabel("Bedrooms")
    axes[0].set_title("Area VS. Bedrooms")

    # Area VS Bathrooms
    sns.scatterplot(data=hb_size_filtered, x="area_m2", y="bathrooms", alpha=0.7, ax=axes[1])
    sns.regplot(data=hb_size_filtered, x="area_m2", y="bathrooms", scatter=False, ax=axes[1], color="black")
    axes[1].set_xlabel("Area (m2)")
    axes[1].set_ylabel("Bathrooms")
    axes[1].set_title("Area VS. Bathrooms")

    st.pyplot(fig)


# -----------------------------
# House Size Distribution by Province
# -----------------------------
def plot_house_size_by_province_t1(hb_size_filtered, hb_location):
    temp = hb_size_filtered.merge(hb_location[["id", "province"]], on="id")
    g = sns.FacetGrid(temp, col="province", col_wrap=4, sharex=False, sharey=False)
    g.map(sns.histplot, "area_m2", kde=True, color=sns.color_palette("Set2")[0])
    g.fig.suptitle("Popular House Size By Province", fontsize=16)
    g.fig.subplots_adjust(top=0.9)
    g.tight_layout()
    st.pyplot(g.fig)


# 4. Timeline Insights (Time Series)

# -----------------------------
# Timeline: Median Price per m2
# -----------------------------
def plot_median_price_trend_t1(hb_timeline):
    # Prepare datetime & price per m2
    hb_timeline["date"] = pd.to_datetime(dict(year=hb_timeline.year, month=hb_timeline.month, day=hb_timeline.day))
    hb_timeline["price_per_m2"] = hb_timeline["price_million_vnd"] / hb_timeline["area_m2"]

    # Filter major provinces & year 2025
    major_provinces = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Bình Dương", "Hưng Yên"]
    hb_filtered = hb_timeline[(hb_timeline["province"].isin(major_provinces)) & (hb_timeline["year"] == 2025)]
    hb_filtered["month"] = hb_filtered["date"].dt.to_period("M").dt.to_timestamp()

    monthly_trend = hb_filtered.groupby(["province", "month"]).agg(
        median_price_m2=("price_per_m2", "median")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=monthly_trend, x="month", y="median_price_m2", hue="province", marker="o", ax=ax)
    ax.set_title("Median Price Per m2 Over Months In Major Provinces")
    ax.set_ylabel("Price (Million VND)")
    ax.set_xlabel("Month")
    plt.legend(loc='upper left')

    st.pyplot(fig)


# -----------------------------
# Timeline: Monthly Listings & Median Price
# -----------------------------
def plot_monthly_listings_and_price_t1(hb_timeline):
    hb_timeline["date"] = pd.to_datetime(dict(year=hb_timeline.year, month=hb_timeline.month, day=hb_timeline.day))
    hb_timeline["price_per_m2"] = hb_timeline["price_million_vnd"] / hb_timeline["area_m2"]

    major_provinces = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Bình Dương", "Hưng Yên"]
    hb_2025 = hb_timeline[(hb_timeline["year"] == 2025) & (hb_timeline["province"].isin(major_provinces))]

    monthly_line = hb_2025.groupby(hb_2025["date"].dt.to_period("M")).agg(
        listing_count=("id", "count"),
        median_price_m2=("price_per_m2", "median")
    ).reset_index()

    monthly_line["month_str"] = monthly_line["date"].dt.to_timestamp().dt.strftime("%Y-%m")

    fig, ax1 = plt.subplots(figsize=(13, 7))
    sns.barplot(data=monthly_line, x="month_str", y="listing_count", color="lightseagreen", ax=ax1)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Number of Listings")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    for i, row in monthly_line.iterrows():
        ax1.text(i, row["listing_count"] + 50, f'{row["listing_count"]}', ha='center', va='bottom', fontsize=9, color='darkgreen')

    ax2 = ax1.twinx()
    sns.lineplot(data=monthly_line, x="month_str", y="median_price_m2", color="salmon", marker="o", ax=ax2)
    ax2.set_ylabel("Median Price Per m2 (Million VND)")

    for i, row in monthly_line.iterrows():
        ax2.text(i, row["median_price_m2"] + 0.2, f'{row["median_price_m2"]:.1f}', ha='center', va='bottom', fontsize=9, color='red')

    plt.title("Monthly Listing Frequency & Median Price Per m2")

    st.pyplot(fig)



# 6. Pre-emergence VS. Post-emergence Comparison

# -----------------------------
# Price per m2: Before vs After Emergence
# -----------------------------
def plot_emergence_price_per_m2_t1(filtered_data):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.violinplot(data=filtered_data, x="province", y="price_per_m2", ax=ax[0], palette="Set2")
    ax[0].set_title("Before Emergence - Price per m2")
    ax[0].tick_params(axis="x", rotation=45)

    sns.violinplot(data=filtered_data, x="province_after_emergence", y="price_per_m2", ax=ax[1], palette="Set3")
    ax[1].set_title("After Emergence - Price per m2")
    ax[1].tick_params(axis="x", rotation=45)

    st.pyplot(fig)


# -----------------------------
# Median price per m2: Before vs After Emergence
# -----------------------------
def plot_median_price_per_m2_emergence_t1(filtered_data):
    median_before = filtered_data.groupby("province")["price_per_m2"].median().reset_index(name="median_price_before")
    median_after = filtered_data.groupby("province_after_emergence")["price_per_m2"].median().reset_index(name="median_price_after")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(data=median_before, x="province", y="median_price_before", ax=ax[0], palette="magma")
    ax[0].set_title("Before Emergence - Median Price per m2")
    ax[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=median_after, x="province_after_emergence", y="median_price_after", ax=ax[1], palette="magma")
    ax[1].set_title("After Emergence - Median Price per m2")
    ax[1].tick_params(axis="x", rotation=45)

    st.pyplot(fig)

# -----------------------------
# Median original price (not per m2): Before vs After Emergence
# -----------------------------
def plot_median_price_emergence_t1(filtered_data):
    median_origin_before = filtered_data.groupby("province")["price_million_vnd"].median().reset_index(name="median_price_origin_before")
    median_origin_after = filtered_data.groupby("province_after_emergence")["price_million_vnd"].median().reset_index(name="median_price_origin_after")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(data=median_origin_before, x="province", y="median_price_origin_before", ax=ax[0], palette="crest")
    ax[0].set_title("Before Emergence - Median Price")
    ax[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=median_origin_after, x="province_after_emergence", y="median_price_origin_after", ax=ax[1], palette="crest")
    ax[1].set_title("After Emergence - Median Price")
    ax[1].tick_params(axis="x", rotation=45)

    st.pyplot(fig)

# Task 2

# -----------------------------
# Location-based Insights
# -----------------------------
def plot_top20_supply_t2(top20_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        y="ward",
        data=top20_df,
        order=top20_df["ward"].value_counts().index,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Top 20 Listing Counts per District / Ward")
    ax.set_xlabel("Number Of Listings")
    ax.set_ylabel("District / Ward")
    
    st.pyplot(fig)

def plot_least20_supply_t2(least20_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        y="ward",
        data=least20_df,
        order=least20_df["ward"].value_counts().index,
        palette="crest",
        ax=ax
    )
    ax.set_title("Least 20 Listing Counts per District / Ward")
    ax.set_xlabel("Number Of Listings")
    ax.set_ylabel("District / Ward")

    st.pyplot(fig)

def plot_listings_per_province_t2(hr_location):
    province_counts = hr_location["province"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x=province_counts.index,
        y=province_counts.values,
        palette="coolwarm",
        ax=ax
    )
    ax.set_title("Listings Count per City / Province")
    ax.set_ylabel("No. Listings")
    ax.set_xlabel("Province")
    plt.xticks(rotation=45)

    st.pyplot(fig)

def plot_median_rent_top20_t2(top20_df):
    median_price = (
        top20_df.groupby("ward")["price_million_vnd"]
        .median()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x=median_price.values,
        y=median_price.index,
        palette="magma",
        ax=ax
    )
    ax.set_title("Median Rent By District / Ward")
    ax.set_xlabel("Median Price (Million VND)")
    ax.set_ylabel("District / Ward")

    st.pyplot(fig)

def plot_median_rent_per_m2_t2(top20_df):
    df = top20_df.copy()
    df["price_per_m2"] = df["price_million_vnd"] / df["area_m2"]

    median_ppm2 = (
        df.groupby("ward")["price_per_m2"]
        .median()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x=median_ppm2.values,
        y=median_ppm2.index,
        palette="plasma",
        ax=ax
    )
    ax.set_title("Median Rent Per m² by District / Ward")
    ax.set_xlabel("Median Price per m² (Million VND/m²)")
    ax.set_ylabel("District / Ward")

    st.pyplot(fig)

# -----------------------------
# Rental Price Structure
# -----------------------------

def plot_price_distribution_t2(rental_dist):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].hist(rental_dist["price_million_vnd"], bins=50, color="green", alpha=0.8)
    ax[0].set_title("Price Distribution (Million VND)")
    ax[0].set_xlabel("Price (Million VND)")
    ax[0].set_ylabel("Count")

    ax[1].hist(
        rental_dist["price_million_vnd"] / rental_dist["area_m2"],
        bins=50,
        color="darkorange",
        alpha=0.8
    )
    ax[1].set_title("Rent per m² Distribution (Million VND/m²)")
    ax[1].set_xlabel("Price per m² (Million VND/m²)")
    ax[1].set_ylabel("Count")

    st.pyplot(fig)


def plot_rent_vs_area_t2(rental_dist):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=rental_dist,
        x="area_m2",
        y="price_million_vnd",
        hue="bedrooms",
        alpha=0.6,
        ax=ax
    )
    sns.regplot(
        data=rental_dist,
        x="area_m2",
        y="price_million_vnd",
        scatter=False,
        color="black",
        lowess=True,
        ax=ax
    )

    ax.set_title("Rent vs Area")
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Price (Million VND)")
    plt.legend(title="Bedrooms", bbox_to_anchor=(1, 1))

    st.pyplot(fig)


def plot_rent_vs_bedrooms_t2(rental_dist):
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(
        x="bedrooms",
        y="price_million_vnd",
        data=rental_dist,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Rent vs Bedrooms")
    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("Price (Million VND)")

    st.pyplot(fig)


def plot_rent_vs_bathrooms_t2(rental_dist):
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(
        x="bathrooms",
        y="price_million_vnd",
        data=rental_dist,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Rent vs Bathrooms")
    ax.set_xlabel("Bathrooms")
    ax.set_ylabel("Price (Million VND)")

    st.pyplot(fig)

# -----------------------------
# House Features & Layout Interpretation
# -----------------------------

def plot_log_area_distribution_t2(rental_size_dist):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(
        np.log1p(rental_size_dist["area_m2"]),
        kde=True,
        bins=30,
        ax=ax
    )
    ax.set_title("Log-transformed Area Distribution (m²)")
    ax.set_xlabel("Log(Area m² + 1)")
    ax.set_ylabel("Count")

    st.pyplot(fig)


def plot_bed_bath_counts_t2(rental_size_dist):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.countplot(x="bedrooms", data=rental_size_dist, palette="Set2", ax=ax[0])
    ax[0].set_title("Number of Bedrooms")

    sns.countplot(x="bathrooms", data=rental_size_dist, palette="Set3", ax=ax[1])
    ax[1].set_title("Number of Bathrooms")

    st.pyplot(fig)


def plot_area_vs_bedrooms_t2(rental_size_dist):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        x="area_m2",
        y="bedrooms",
        hue="floors",
        data=rental_size_dist,
        alpha=0.6,
        ax=ax
    )
    ax.set_title("Area vs Bedrooms")
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Bedrooms")
    plt.legend(title="Floors", bbox_to_anchor=(1, 1))

    st.pyplot(fig)


def plot_area_vs_bathrooms_t2(rental_size_dist):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        x="area_m2",
        y="bathrooms",
        hue="floors",
        data=rental_size_dist,
        alpha=0.6,
        ax=ax
    )
    ax.set_title("Area vs Bathrooms")
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Bathrooms")
    plt.legend(title="Floors", bbox_to_anchor=(1, 1))

    st.pyplot(fig)


def plot_floors_frontage_t2(rental_location_list):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.countplot(x="floors", data=rental_location_list, palette="icefire", ax=ax[0])
    ax[0].set_title("Number of Floors")

    sns.countplot(x="frontage", data=rental_location_list, palette="twilight", ax=ax[1])
    ax[1].set_title("Frontage Presence")

    st.pyplot(fig)

# -----------------------------
# Date / Listing Dynamicss
# -----------------------------

def plot_listing_count_by_month_t2(sample_timeline):
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.countplot(
        x="month",
        data=sample_timeline,
        palette="coolwarm",
        ax=ax
    )

    ax.set_title("Listings Count by Month (Year 2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Listings")

    st.pyplot(fig)

# Ho Chi Minh City
def plot_hcm_listings_by_ward_t2(sample_timeline):
    hcm_df = sample_timeline[sample_timeline["province"] == "Hồ Chí Minh"]

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.countplot(
        data=hcm_df,
        y="ward",
        order=hcm_df["ward"].value_counts().index,
        palette="Spectral",
        ax=ax
    )

    ax.set_title("Ho Chi Minh City – Number of Rental Listings by Ward")
    ax.set_xlabel("Count")
    ax.set_ylabel("Ward / District")

    st.pyplot(fig)

# Ha Noi
def plot_hanoi_listings_by_ward_t2(sample_timeline):
    hn_df = sample_timeline[sample_timeline["province"] == "Hà Nội"]

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.countplot(
        data=hn_df,
        y="ward",
        order=hn_df["ward"].value_counts().index,
        palette="Paired",
        ax=ax
    )

    ax.set_title("Ha Noi – Number of Rental Listings by Ward")
    ax.set_xlabel("Count")
    ax.set_ylabel("Ward / District")

    st.pyplot(fig)

# Da Nang
def plot_danang_listings_by_month_t2(danang_df):
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.countplot(
        data=danang_df,
        x="month",
        order=sorted(danang_df["month"].unique()),
        palette="muted",
        ax=ax
    )

    ax.set_title("Da Nang – Rental Listing Counts per Month (2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("No. Listings")

    st.pyplot(fig)

def plot_danang_monthly_trend_t2(monthly_counts):
    fig, ax = plt.subplots(figsize=(8, 5))

    monthly_counts.plot(
        kind="line",
        marker="o",
        color="red",
        ax=ax
    )

    ax.set_title("Da Nang – Monthly Listings Trend (2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("No. Listings")
    ax.grid(True)

    st.pyplot(fig)

