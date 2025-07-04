import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/admin/Documents/Oasis Infobyte/Cleaning Data/AB_NYC_2019.csv')

print("NYC Airbnb Data Cleaning Project")

print("\n1. INITIAL DATA EXPLORATION")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nData types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())

print("\n2. DATA INTEGRITY ASSESSMENT")

print("Missing values per column:")
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percentage
}).sort_values('Missing Percentage', ascending=False)
print(missing_summary[missing_summary['Missing Count'] > 0])

print("\nBasic statistics for numerical columns:")
print(df.describe())

print("\n3. MISSING DATA HANDLING")

df_clean = df.copy()

df_clean['host_name'] = df_clean['host_name'].fillna('Unknown Host')

df_clean['name'] = df_clean['name'].fillna('Unnamed Listing')

df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)

print("After handling missing values:")
print(df_clean.isnull().sum())

print("\n4. DUPLICATE REMOVAL")

print(f"Original dataset size: {len(df_clean)}")
print(f"Duplicate rows: {df_clean.duplicated().sum()}")

df_clean = df_clean.drop_duplicates()
print(f"After removing duplicates: {len(df_clean)}")

duplicate_check = df_clean.duplicated(subset=['host_id', 'latitude', 'longitude', 'room_type'])
print(f"Potential duplicate listings (same host, location, room type): {duplicate_check.sum()}")

print("\n5. STANDARDIZATION")

df_clean['neighbourhood_group'] = df_clean['neighbourhood_group'].str.strip().str.title()
df_clean['neighbourhood'] = df_clean['neighbourhood'].str.strip().str.title()
df_clean['room_type'] = df_clean['room_type'].str.strip().str.title()

df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce')

df_clean['last_review'] = pd.to_datetime(df_clean['last_review'], errors='coerce')

df_clean['days_since_last_review'] = (pd.Timestamp.now() - df_clean['last_review']).dt.days
df_clean['has_reviews'] = df_clean['number_of_reviews'] > 0

print("Standardization complete!")
print(f"Unique neighbourhood groups: {df_clean['neighbourhood_group'].unique()}")
print(f"Unique room types: {df_clean['room_type'].unique()}")

print("\n6. OUTLIER DETECTION AND HANDLING")

def identify_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

price_outliers = identify_outliers(df_clean, 'price')
print(f"Price outliers (IQR method): {len(price_outliers)}")
print(f"Price range: ${df_clean['price'].min()} - ${df_clean['price'].max()}")

nights_outliers = identify_outliers(df_clean, 'minimum_nights')
print(f"Minimum nights outliers: {len(nights_outliers)}")
print(f"Minimum nights range: {df_clean['minimum_nights'].min()} - {df_clean['minimum_nights'].max()}")

df_clean = df_clean[(df_clean['price'] > 0) & (df_clean['price'] <= 1000)]
print(f"After removing extreme price outliers: {len(df_clean)} rows")

df_clean = df_clean[df_clean['minimum_nights'] <= 365]
print(f"After removing extreme minimum nights: {len(df_clean)} rows")

print("\n7. DATA VALIDATION")

nyc_lat_range = (40.4, 40.9)
nyc_lon_range = (-74.3, -73.7)

invalid_coords = df_clean[
    (df_clean['latitude'] < nyc_lat_range[0]) | 
    (df_clean['latitude'] > nyc_lat_range[1]) |
    (df_clean['longitude'] < nyc_lon_range[0]) | 
    (df_clean['longitude'] > nyc_lon_range[1])
]
print(f"Invalid coordinates: {len(invalid_coords)}")

df_clean = df_clean[
    (df_clean['latitude'] >= nyc_lat_range[0]) & 
    (df_clean['latitude'] <= nyc_lat_range[1]) &
    (df_clean['longitude'] >= nyc_lon_range[0]) & 
    (df_clean['longitude'] <= nyc_lon_range[1])
]

inconsistent_reviews = df_clean[
    (df_clean['number_of_reviews'] == 0) & (df_clean['reviews_per_month'] > 0)
]
print(f"Inconsistent review data: {len(inconsistent_reviews)}")

print("\n8. FINAL DATA QUALITY REPORT")

print(f"Original dataset: {len(df)} rows")
print(f"Cleaned dataset: {len(df_clean)} rows")
print(f"Rows removed: {len(df) - len(df_clean)} ({((len(df) - len(df_clean)) / len(df)) * 100:.2f}%)")

print("\nFinal data quality metrics:")
print(f"- Missing values: {df_clean.isnull().sum().sum()}")
print(f"- Duplicate rows: {df_clean.duplicated().sum()}")
print(f"- Invalid coordinates: 0")
print(f"- Zero prices: {(df_clean['price'] == 0).sum()}")

print("\n9. SAVING CLEANED DATA")

df_clean.to_csv('C:/Users/admin/Documents/Oasis Infobyte/Cleaning Data/NYC_Airbnb_Cleaned.csv', index=False)
print("Cleaned dataset saved as 'NYC_Airbnb_Cleaned.csv'")

quality_report = {
    'Original_Rows': len(df),
    'Cleaned_Rows': len(df_clean),
    'Rows_Removed': len(df) - len(df_clean),
    'Removal_Percentage': ((len(df) - len(df_clean)) / len(df)) * 100,
    'Missing_Values_Remaining': df_clean.isnull().sum().sum(),
    'Duplicate_Rows': df_clean.duplicated().sum(),
    'Price_Range': f"${df_clean['price'].min()} - ${df_clean['price'].max()}",
    'Unique_Neighbourhoods': df_clean['neighbourhood'].nunique(),
    'Unique_Hosts': df_clean['host_id'].nunique()
}

print("\nCleaning Summary Report:")
for key, value in quality_report.items():
    print(f"{key.replace('_', ' ')}: {value}")

print("\n10. CREATING VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].hist(df['price'], bins=50, alpha=0.7, label='Original', color='red')
axes[0, 0].hist(df_clean['price'], bins=50, alpha=0.7, label='Cleaned', color='blue')
axes[0, 0].set_title('Price Distribution: Before vs After Cleaning')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 500)

missing_original = df.isnull().sum().sum()
missing_cleaned = df_clean.isnull().sum().sum()
axes[0, 1].bar(['Original', 'Cleaned'], [missing_original, missing_cleaned], 
               color=['red', 'blue'], alpha=0.7)
axes[0, 1].set_title('Total Missing Values: Before vs After')
axes[0, 1].set_ylabel('Number of Missing Values')

room_type_counts = df_clean['room_type'].value_counts()
axes[1, 0].pie(room_type_counts.values, labels=room_type_counts.index, autopct='%1.1f%%')
axes[1, 0].set_title('Room Type Distribution (Cleaned Data)')

borough_counts = df_clean['neighbourhood_group'].value_counts()
axes[1, 1].bar(borough_counts.index, borough_counts.values, color='skyblue')
axes[1, 1].set_title('Listings by Borough (Cleaned Data)')
axes[1, 1].set_ylabel('Number of Listings')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('C:/Users/admin/Documents/Oasis Infobyte/Cleaning Data/data_cleaning_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nData cleaning complete! Check 'data_cleaning_results.png' for visualizations.")
print("The cleaned dataset is ready for further analysis and modeling.")