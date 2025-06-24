

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("McDonald's Nutrition Facts - Exploratory Data Analysis")

# Load the data
df = pd.read_csv('C:\\Users\\admin\\Documents\\Oasis Infobyte\\EDA on Retail Sales Data\\menu.csv')


print(" BASIC DATA EXPLORATION")


# Display basic information
print(f"\n Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:")
print(df.dtypes)

# Display first few rows
print(f"\n First 5 rows:")
print(df.head())

# Check for missing values
print(f"\n Missing values:")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print(" No missing values found!")
else:
    print(missing_values[missing_values > 0])

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n Duplicate rows: {duplicates}")

# Basic statistics for numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(f"\n Numerical columns: {len(numerical_cols)}")
print(list(numerical_cols))

categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\n Categorical columns: {len(categorical_cols)}")
print(list(categorical_cols))


print(" DESCRIPTIVE STATISTICS")


# Summary statistics for key nutrition metrics
key_nutrition_cols = ['Calories', 'Total Fat', 'Saturated Fat', 'Cholesterol', 
                     'Sodium', 'Carbohydrates', 'Sugars', 'Protein']

print("\n Key Nutrition Metrics Summary:")
if all(col in df.columns for col in key_nutrition_cols):
    print(df[key_nutrition_cols].describe().round(2))
else:
    # Use available numerical columns
    available_nutrition_cols = [col for col in key_nutrition_cols if col in df.columns]
    if available_nutrition_cols:
        print(df[available_nutrition_cols].describe().round(2))
    else:
        print(df[numerical_cols].describe().round(2))

# Category analysis
if 'Category' in df.columns:
    print(f"\n Menu Categories:")
    category_counts = df['Category'].value_counts()
    print(category_counts)
    print(f"\nTotal categories: {df['Category'].nunique()}")


print(" MENU CATEGORY ANALYSIS")


if 'Category' in df.columns:
    # Category distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('McDonald\'s Menu Category Analysis', fontsize=16, fontweight='bold')
    
    # Category distribution pie chart
    category_counts = df['Category'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
    axes[0,0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', colors=colors)
    axes[0,0].set_title('Distribution of Menu Items by Category')
    
    # Average calories by category
    if 'Calories' in df.columns:
        avg_calories_by_category = df.groupby('Category')['Calories'].mean().sort_values(ascending=False)
        bars = axes[0,1].bar(range(len(avg_calories_by_category)), avg_calories_by_category.values, color='orange')
        axes[0,1].set_title('Average Calories by Category')
        axes[0,1].set_ylabel('Average Calories')
        axes[0,1].set_xticks(range(len(avg_calories_by_category)))
        axes[0,1].set_xticklabels(avg_calories_by_category.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_calories_by_category.values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                          f'{value:.0f}', ha='center', va='bottom')
    
    # Total fat by category
    if 'Total Fat' in df.columns:
        avg_fat_by_category = df.groupby('Category')['Total Fat'].mean().sort_values(ascending=False)
        axes[1,0].bar(range(len(avg_fat_by_category)), avg_fat_by_category.values, color='red')
        axes[1,0].set_title('Average Total Fat by Category')
        axes[1,0].set_ylabel('Average Total Fat (g)')
        axes[1,0].set_xticks(range(len(avg_fat_by_category)))
        axes[1,0].set_xticklabels(avg_fat_by_category.index, rotation=45, ha='right')
    
    # Protein by category
    if 'Protein' in df.columns:
        avg_protein_by_category = df.groupby('Category')['Protein'].mean().sort_values(ascending=False)
        axes[1,1].bar(range(len(avg_protein_by_category)), avg_protein_by_category.values, color='green')
        axes[1,1].set_title('Average Protein by Category')
        axes[1,1].set_ylabel('Average Protein (g)')
        axes[1,1].set_xticks(range(len(avg_protein_by_category)))
        axes[1,1].set_xticklabels(avg_protein_by_category.index, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Print category insights
    print(f"\n Category Insights:")
    if 'Calories' in df.columns:
        highest_cal_category = avg_calories_by_category.index[0]
        lowest_cal_category = avg_calories_by_category.index[-1]
        print(f"Highest calorie category: {highest_cal_category} ({avg_calories_by_category.iloc[0]:.0f} calories)")
        print(f"Lowest calorie category: {lowest_cal_category} ({avg_calories_by_category.iloc[-1]:.0f} calories)")


print(" NUTRITIONAL CONTENT ANALYSIS")


# Distribution of key nutritional components
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribution of Key Nutritional Components', fontsize=16, fontweight='bold')

# Calories distribution
if 'Calories' in df.columns:
    axes[0,0].hist(df['Calories'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Distribution of Calories')
    axes[0,0].set_xlabel('Calories')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(df['Calories'].mean(), color='red', linestyle='--', label=f'Mean: {df["Calories"].mean():.0f}')
    axes[0,0].legend()

# Total Fat distribution
if 'Total Fat' in df.columns:
    axes[0,1].hist(df['Total Fat'], bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[0,1].set_title('Distribution of Total Fat')
    axes[0,1].set_xlabel('Total Fat (g)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].axvline(df['Total Fat'].mean(), color='red', linestyle='--', label=f'Mean: {df["Total Fat"].mean():.1f}g')
    axes[0,1].legend()

# Sodium distribution
if 'Sodium' in df.columns:
    axes[1,0].hist(df['Sodium'], bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Distribution of Sodium')
    axes[1,0].set_xlabel('Sodium (mg)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].axvline(df['Sodium'].mean(), color='darkred', linestyle='--', label=f'Mean: {df["Sodium"].mean():.0f}mg')
    axes[1,0].legend()

# Protein distribution
if 'Protein' in df.columns:
    axes[1,1].hist(df['Protein'], bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Distribution of Protein')
    axes[1,1].set_xlabel('Protein (g)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].axvline(df['Protein'].mean(), color='darkgreen', linestyle='--', label=f'Mean: {df["Protein"].mean():.1f}g')
    axes[1,1].legend()

plt.tight_layout()
plt.show()


print(" CORRELATION ANALYSIS")


# Select numerical columns for correlation analysis
correlation_cols = [col for col in numerical_cols if not col.endswith('(% Daily Value)')][:10]  # Limit to first 10 for readability

if len(correlation_cols) >= 2:
    correlation_matrix = df[correlation_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Nutritional Components')
    plt.tight_layout()
    plt.show()
    
    # Print strong correlations
    print(f"\n Strong Correlations (|r| > 0.7):")
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                strong_correlations.append((col1, col2, corr_val))
                print(f"  {col1} & {col2}: {corr_val:.3f}")
    
    if not strong_correlations:
        print("  No strong correlations found (|r| > 0.7)")


print(" HIGH-CALORIE VS HEALTHY OPTIONS ANALYSIS")


if 'Calories' in df.columns:
    # Define calorie thresholds
    high_calorie_threshold = df['Calories'].quantile(0.75)  # Top 25%
    low_calorie_threshold = df['Calories'].quantile(0.25)   # Bottom 25%
    
    print(f" High-calorie threshold (top 25%): {high_calorie_threshold:.0f} calories")
    print(f" Low-calorie threshold (bottom 25%): {low_calorie_threshold:.0f} calories")
    
    # High-calorie items
    high_calorie_items = df[df['Calories'] >= high_calorie_threshold].sort_values('Calories', ascending=False)
    print(f"\n TOP 10 HIGHEST CALORIE ITEMS:")
    for i, (idx, item) in enumerate(high_calorie_items.head(10).iterrows(), 1):
        item_name = item['Item'] if 'Item' in item else f"Item {idx}"
        print(f"  {i:2d}. {item_name}: {item['Calories']:.0f} calories")
    
    # Low-calorie items
    low_calorie_items = df[df['Calories'] <= low_calorie_threshold].sort_values('Calories')
    print(f"\n TOP 10 LOWEST CALORIE ITEMS:")
    for i, (idx, item) in enumerate(low_calorie_items.head(10).iterrows(), 1):
        item_name = item['Item'] if 'Item' in item else f"Item {idx}"
        print(f"  {i:2d}. {item_name}: {item['Calories']:.0f} calories")



print(" MACRONUTRIENT ANALYSIS")


# Calculate macronutrient ratios
macro_cols = ['Total Fat', 'Carbohydrates', 'Protein']
available_macro_cols = [col for col in macro_cols if col in df.columns]

if len(available_macro_cols) >= 2:
    # Create macronutrient composition visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Macronutrient Analysis', fontsize=16, fontweight='bold')
    
    # Average macronutrient composition
    avg_macros = df[available_macro_cols].mean()
    colors = ['red', 'blue', 'green'][:len(available_macro_cols)]
    
    axes[0].pie(avg_macros.values, labels=avg_macros.index, autopct='%1.1f%%', colors=colors)
    axes[0].set_title('Average Macronutrient Composition (by weight)')
    
    # Macronutrient scatter plot (if we have at least 2 macros)
    if len(available_macro_cols) >= 2:
        scatter = axes[1].scatter(df[available_macro_cols[0]], df[available_macro_cols[1]], 
                                 c=df['Calories'] if 'Calories' in df.columns else 'blue', 
                                 cmap='YlOrRd', alpha=0.6, s=60)
        axes[1].set_xlabel(f'{available_macro_cols[0]} (g)')
        axes[1].set_ylabel(f'{available_macro_cols[1]} (g)')
        axes[1].set_title(f'{available_macro_cols[0]} vs {available_macro_cols[1]}')
        if 'Calories' in df.columns:
            plt.colorbar(scatter, ax=axes[1], label='Calories')
    
    plt.tight_layout()
    plt.show()
    
    # Print macronutrient insights
    print(f"\n Macronutrient Insights:")
    for col in available_macro_cols:
        print(f"  Average {col}: {df[col].mean():.1f}g")
        print(f"  Range {col}: {df[col].min():.1f}g - {df[col].max():.1f}g")


print(" SODIUM AND SUGAR ANALYSIS (HEALTH CONCERNS)")


# Sodium analysis
if 'Sodium' in df.columns:
    # Recommended daily sodium intake is 2300mg
    recommended_sodium = 2300
    high_sodium_items = df[df['Sodium'] > recommended_sodium * 0.3]  # >30% of daily value
    
    print(f" HIGH SODIUM ITEMS (>30% of daily recommended intake):")
    print(f"   Daily recommended sodium: {recommended_sodium}mg")
    print(f"   High sodium threshold: {recommended_sodium * 0.3:.0f}mg")
    print(f"   Number of high sodium items: {len(high_sodium_items)}")
    
    if len(high_sodium_items) > 0:
        high_sodium_sorted = high_sodium_items.sort_values('Sodium', ascending=False)
        print(f"\n   Top 5 highest sodium items:")
        for i, (idx, item) in enumerate(high_sodium_sorted.head(5).iterrows(), 1):
            item_name = item['Item'] if 'Item' in item else f"Item {idx}"
            print(f"   {i}. {item_name}: {item['Sodium']:.0f}mg")

# Sugar analysis
if 'Sugars' in df.columns:
    # WHO recommends <25g of added sugars per day
    recommended_sugar = 25
    high_sugar_items = df[df['Sugars'] > recommended_sugar * 0.5]  # >50% of recommended
    
    print(f"\n HIGH SUGAR ITEMS (>50% of WHO recommended daily intake):")
    print(f"   WHO recommended added sugars: <{recommended_sugar}g per day")
    print(f"   High sugar threshold: {recommended_sugar * 0.5:.0f}g")
    print(f"   Number of high sugar items: {len(high_sugar_items)}")
    
    if len(high_sugar_items) > 0:
        high_sugar_sorted = high_sugar_items.sort_values('Sugars', ascending=False)
        print(f"\n   Top 5 highest sugar items:")
        for i, (idx, item) in enumerate(high_sugar_sorted.head(5).iterrows(), 1):
            item_name = item['Item'] if 'Item' in item else f"Item {idx}"
            print(f"   {i}. {item_name}: {item['Sugars']:.1f}g")


print(" ADVANCED VISUALIZATIONS")


# Create comprehensive nutrition dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('McDonald\'s Menu - Comprehensive Nutrition Dashboard', fontsize=16, fontweight='bold')

# 1. Calorie vs Fat scatter plot
if 'Calories' in df.columns and 'Total Fat' in df.columns:
    if 'Category' in df.columns:
        categories = df['Category'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        for i, category in enumerate(categories):
            category_data = df[df['Category'] == category]
            axes[0,0].scatter(category_data['Total Fat'], category_data['Calories'], 
                            c=[colors[i]], label=category, alpha=0.7)
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[0,0].scatter(df['Total Fat'], df['Calories'], alpha=0.7)
    
    axes[0,0].set_xlabel('Total Fat (g)')
    axes[0,0].set_ylabel('Calories')
    axes[0,0].set_title('Calories vs Total Fat by Category')

# 2. Box plot of calories by category
if 'Category' in df.columns and 'Calories' in df.columns:
    df.boxplot(column='Calories', by='Category', ax=axes[0,1])
    axes[0,1].set_title('Calorie Distribution by Category')
    axes[0,1].set_xlabel('Category')
    axes[0,1].set_ylabel('Calories')
    axes[0,1].tick_params(axis='x', rotation=45)

# 3. Protein content analysis
if 'Protein' in df.columns:
    axes[0,2].hist(df['Protein'], bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[0,2].set_title('Distribution of Protein Content')
    axes[0,2].set_xlabel('Protein (g)')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].axvline(df['Protein'].mean(), color='red', linestyle='--', 
                     label=f'Mean: {df["Protein"].mean():.1f}g')
    axes[0,2].legend()

# 4. Sodium vs Calories
if 'Sodium' in df.columns and 'Calories' in df.columns:
    axes[1,0].scatter(df['Sodium'], df['Calories'], alpha=0.6, color='red')
    axes[1,0].set_xlabel('Sodium (mg)')
    axes[1,0].set_ylabel('Calories')
    axes[1,0].set_title('Sodium vs Calories')

# 5. Sugar content by category
if 'Sugars' in df.columns and 'Category' in df.columns:
    sugar_by_category = df.groupby('Category')['Sugars'].mean().sort_values(ascending=False)
    bars = axes[1,1].bar(range(len(sugar_by_category)), sugar_by_category.values, color='pink')
    axes[1,1].set_title('Average Sugar Content by Category')
    axes[1,1].set_ylabel('Average Sugars (g)')
    axes[1,1].set_xticks(range(len(sugar_by_category)))
    axes[1,1].set_xticklabels(sugar_by_category.index, rotation=45, ha='right')

# 6. Nutritional balance radar chart concept (simplified bar chart)
if len(available_macro_cols) >= 3:
    avg_macros_normalized = df[available_macro_cols].mean() / df[available_macro_cols].mean().max()
    axes[1,2].bar(available_macro_cols, avg_macros_normalized, color=['red', 'blue', 'green'])
    axes[1,2].set_title('Normalized Average Macronutrients')
    axes[1,2].set_ylabel('Normalized Value')
    axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()