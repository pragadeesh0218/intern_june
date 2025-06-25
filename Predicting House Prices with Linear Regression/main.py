
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HousePricePrediction:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, file_path):
        """Load and return the dataset"""
        try:
            self.df = pd.read_csv("C:Users/admin/Documents/Oasis Infobyte/Predicting House Prices with Linear Regression/Housing.csv")
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create sample data if file not found
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample housing data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic housing data
        data = {
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'sqft_living': np.random.randint(500, 5000, n_samples),
            'sqft_lot': np.random.randint(1000, 20000, n_samples),
            'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
            'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'view': np.random.randint(0, 5, n_samples),
            'condition': np.random.randint(1, 6, n_samples),
            'grade': np.random.randint(3, 13, n_samples),
            'yr_built': np.random.randint(1900, 2015, n_samples),
            'zipcode': np.random.choice([98001, 98002, 98003, 98004, 98005], n_samples)
        }
        
        # Create realistic price based on features
        price = (
            data['bedrooms'] * 15000 +
            data['bathrooms'] * 20000 +
            data['sqft_living'] * 100 +
            data['sqft_lot'] * 2 +
            data['floors'] * 10000 +
            data['waterfront'] * 200000 +
            data['view'] * 25000 +
            data['condition'] * 15000 +
            data['grade'] * 30000 +
            (2015 - data['yr_built']) * -500 +
            np.random.normal(0, 50000, n_samples)
        )
        
        data['price'] = np.maximum(price, 50000)  # Minimum price
        
        self.df = pd.DataFrame(data)
        print("Sample dataset created for demonstration")
        print(f"Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Comprehensive data exploration"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print("\n1. DATASET OVERVIEW:")
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n2. DATA TYPES:")
        print(self.df.dtypes)
        
        print("\n3. MISSING VALUES:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found!")
        
        print("\n4. BASIC STATISTICS:")
        print(self.df.describe())
        
        # Visualizations
        self.create_exploration_plots()
        
        return self.df.info()
    
    def create_exploration_plots(self):
        """Create comprehensive exploration visualizations"""
        # Set up the plotting area
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Price distribution
        plt.subplot(3, 4, 1)
        plt.hist(self.df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        
        # 2. Price vs key features
        key_features = ['bedrooms', 'bathrooms', 'sqft_living', 'grade']
        for i, feature in enumerate(key_features, 2):
            plt.subplot(3, 4, i)
            plt.scatter(self.df[feature], self.df['price'], alpha=0.6)
            plt.title(f'Price vs {feature.title()}')
            plt.xlabel(feature.title())
            plt.ylabel('Price')
        
        # 6. Correlation heatmap
        plt.subplot(3, 4, 6)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix')
        
        # 7. Box plot for categorical features
        if 'condition' in self.df.columns:
            plt.subplot(3, 4, 7)
            sns.boxplot(x='condition', y='price', data=self.df)
            plt.title('Price by Condition')
            plt.xticks(rotation=45)
        
        # 8. Price by year built
        if 'yr_built' in self.df.columns:
            plt.subplot(3, 4, 8)
            plt.scatter(self.df['yr_built'], self.df['price'], alpha=0.6)
            plt.title('Price vs Year Built')
            plt.xlabel('Year Built')
            plt.ylabel('Price')
        
        # 9-12. Additional plots for key relationships
        remaining_features = ['floors', 'view', 'waterfront', 'sqft_lot']
        for i, feature in enumerate(remaining_features, 9):
            if feature in self.df.columns:
                plt.subplot(3, 4, i)
                if feature == 'waterfront':
                    sns.boxplot(x=feature, y='price', data=self.df)
                else:
                    plt.scatter(self.df[feature], self.df['price'], alpha=0.6)
                plt.title(f'Price vs {feature.title()}')
                plt.xlabel(feature.title())
                plt.ylabel('Price')
        
        plt.tight_layout()
        plt.show()
    
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\n" + "="*50)
        print("DATA CLEANING & PREPROCESSING")
        print("="*50)
        
        # Make a copy for processing
        self.df_clean = self.df.copy()
        
        # Handle missing values
        numeric_columns = self.df_clean.select_dtypes(include=[np.number]).columns
        categorical_columns = self.df_clean.select_dtypes(exclude=[np.number]).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if self.df_clean[col].isnull().sum() > 0:
                self.df_clean[col].fillna(self.df_clean[col].median(), inplace=True)
                print(f"Filled {col} missing values with median")
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if self.df_clean[col].isnull().sum() > 0:
                self.df_clean[col].fillna(self.df_clean[col].mode()[0], inplace=True)
                print(f"Filled {col} missing values with mode")
        
        # Remove outliers (optional - using IQR method for price)
        Q1 = self.df_clean['price'].quantile(0.25)
        Q3 = self.df_clean['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_removed = len(self.df_clean) - len(self.df_clean[
            (self.df_clean['price'] >= lower_bound) & 
            (self.df_clean['price'] <= upper_bound)
        ])
        
        print(f"Identified {outliers_removed} price outliers")
        print("Keeping outliers for this demonstration")
        
        # Encode categorical variables
        for col in categorical_columns:
            if col != 'price':  # Don't encode target variable
                le = LabelEncoder()
                self.df_clean[col] = le.fit_transform(self.df_clean[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded categorical variable: {col}")
        
        print(f"\nCleaned dataset shape: {self.df_clean.shape}")
        return self.df_clean
    
    def feature_selection(self, k=10):
        """Select the best features for the model"""
        print("\n" + "="*50)
        print("FEATURE SELECTION")
        print("="*50)
        
        # Separate features and target
        X = self.df_clean.drop('price', axis=1)
        y = self.df_clean['price']
        
        # Statistical feature selection
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        self.feature_names = selected_features
        
        print(f"Selected {len(selected_features)} best features:")
        
        # Show feature scores
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_,
            'Selected': selector.get_support()
        }).sort_values('Score', ascending=False)
        
        print(feature_scores.head(10))
        
        return X[selected_features], y
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train the linear regression model"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Get features and target
        X, y = self.feature_selection()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print("Model training completed!")
        
        # Show model coefficients
        coefficients = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print("\nFeature Coefficients (sorted by absolute value):")
        print(coefficients)
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        self.y_train_pred = self.model.predict(self.X_train_scaled)
        self.y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        test_mse = mean_squared_error(self.y_test, self.y_test_pred)
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        
        # Display results
        print("TRAINING SET PERFORMANCE:")
        print(f"MSE: ${train_mse:,.2f}")
        print(f"RMSE: ${train_rmse:,.2f}")
        print(f"MAE: ${train_mae:,.2f}")
        print(f"R²: {train_r2:.4f}")
        
        print("\nTEST SET PERFORMANCE:")
        print(f"MSE: ${test_mse:,.2f}")
        print(f"RMSE: ${test_rmse:,.2f}")
        print(f"MAE: ${test_mae:,.2f}")
        print(f"R²: {test_r2:.4f}")
        
        # Check for overfitting
        if train_r2 - test_r2 > 0.1:
            print(f"\n  Warning: Possible overfitting detected!")
            print(f"Training R² - Test R² = {train_r2 - test_r2:.4f}")
        else:
            print(f"\n Good model generalization!")
        
        return {
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2
        }
    
    def create_evaluation_plots(self):
        """Create comprehensive evaluation visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Actual vs Predicted (Training)
        axes[0, 0].scatter(self.y_train, self.y_train_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([self.y_train.min(), self.y_train.max()], 
                       [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price')
        axes[0, 0].set_ylabel('Predicted Price')
        axes[0, 0].set_title('Training Set: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted (Test)
        axes[0, 1].scatter(self.y_test, self.y_test_pred, alpha=0.6, color='green')
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Price')
        axes[0, 1].set_ylabel('Predicted Price')
        axes[0, 1].set_title('Test Set: Actual vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals (Training)
        train_residuals = self.y_train - self.y_train_pred
        axes[0, 2].scatter(self.y_train_pred, train_residuals, alpha=0.6, color='blue')
        axes[0, 2].axhline(y=0, color='r', linestyle='--')
        axes[0, 2].set_xlabel('Predicted Price')
        axes[0, 2].set_ylabel('Residuals')
        axes[0, 2].set_title('Training Set: Residual Plot')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Residuals (Test)
        test_residuals = self.y_test - self.y_test_pred
        axes[1, 0].scatter(self.y_test_pred, test_residuals, alpha=0.6, color='green')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Price')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Test Set: Residual Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Residuals Distribution
        axes[1, 1].hist(test_residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature Importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(self.model.coef_)
        }).sort_values('Importance', ascending=True)
        
        axes[1, 2].barh(range(len(feature_importance)), feature_importance['Importance'])
        axes[1, 2].set_yticks(range(len(feature_importance)))
        axes[1, 2].set_yticklabels(feature_importance['Feature'])
        axes[1, 2].set_xlabel('Absolute Coefficient Value')
        axes[1, 2].set_title('Feature Importance')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def make_predictions(self, new_data=None):
        """Make predictions on new data"""
        if new_data is None:
            # Create sample predictions
            sample_indices = np.random.choice(len(self.X_test), 5, replace=False)
            sample_data = self.X_test.iloc[sample_indices]
            actual_prices = self.y_test.iloc[sample_indices]
        else:
            sample_data = new_data
            actual_prices = None
        
        # Scale the data
        sample_scaled = self.scaler.transform(sample_data)
        predictions = self.model.predict(sample_scaled)
        
        print("\n" + "="*50)
        print("SAMPLE PREDICTIONS")
        print("="*50)
        
        results = pd.DataFrame({
            'Predicted_Price': predictions,
        })
        
        if actual_prices is not None:
            results['Actual_Price'] = actual_prices.values
            results['Difference'] = results['Actual_Price'] - results['Predicted_Price']
            results['Accuracy_%'] = 100 * (1 - np.abs(results['Difference']) / results['Actual_Price'])
        
        # Add feature values
        for i, feature in enumerate(self.feature_names):
            results[feature] = sample_data.iloc[:, i].values
        
        print(results.round(2))
        return results
    
    def run_complete_analysis(self, file_path=None):
        """Run the complete house price prediction analysis"""
        print(" HOUSE PRICE PREDICTION WITH LINEAR REGRESSION")
        
        
        # 1. Load data
        if file_path:
            self.load_data(file_path)
        else:
            self.create_sample_data()
        
        # 2. Explore data
        self.explore_data()
        
        # 3. Clean data
        self.clean_data()
        
        # 4. Train model
        self.train_model()
        
        # 5. Evaluate model
        metrics = self.evaluate_model()
        
        # 6. Create visualizations
        self.create_evaluation_plots()
        
        # 7. Make sample predictions
        self.make_predictions()
        
        
        print(" ANALYSIS COMPLETE!")
        
        
        return metrics

# Example usage and main execution
if __name__ == "__main__":
    # Initialize the predictor
    predictor = HousePricePrediction()
    
    # Option 1: Use with your own dataset
    # metrics = predictor.run_complete_analysis('your_housing_data.csv')
    
    # Option 2: Use with sample data (for demonstration)
    print("Running analysis with sample data...")
    metrics = predictor.run_complete_analysis()
    
    # Additional analysis examples
    
    print("ADDITIONAL INSIGHTS")
    
    
    # Feature correlation with price
    if hasattr(predictor, 'df_clean'):
        price_corr = predictor.df_clean.corr()['price'].sort_values(ascending=False)
        print("\nTop features correlated with price:")
        print(price_corr.head(10))
    
    print("\n Key Takeaways:")
    print("- Linear regression provides interpretable results")
    print("- Feature selection helps improve model performance")
    print("- Regular evaluation prevents overfitting")
    print("- Visualization helps understand model behavior")
    print("\n Next Steps:")
    print("- Try polynomial features for non-linear relationships")
    print("- Experiment with regularization (Ridge/Lasso)")
    print("- Consider ensemble methods for better accuracy")
    print("- Validate assumptions of linear regression")