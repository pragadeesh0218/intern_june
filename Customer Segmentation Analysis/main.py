
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

class CustomerSegmentationAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.optimal_clusters = None
        self.customer_segments = None
        self.rfm_data = None
        
    def load_data(self, file_path=None):
        """Load customer data from file or create sample data"""
        if file_path:
            try:
                self.df = pd.read_csv("C:/Users/admin/Documents/Oasis Infobyte/Customer Segmentation Analysis/ifood_df.csv")
                print(f" Dataset loaded successfully!")
                print(f"Shape: {self.df.shape}")
                return self.df
            except Exception as e:
                print(f" Error loading data: {e}")
                print("Creating sample data for demonstration...")
                return self.create_sample_data()
        else:
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create comprehensive sample e-commerce customer data"""
        np.random.seed(42)
        n_customers = 2000
        
        # Customer demographics
        customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
        
        # Generate realistic customer segments for ground truth
        segment_sizes = [0.3, 0.25, 0.2, 0.15, 0.1]  # High-value, Regular, Occasional, New, Churned
        segments = np.random.choice(['High-Value', 'Regular', 'Occasional', 'New', 'Churned'], 
                                  n_customers, p=segment_sizes)
        
        data = {
            'CustomerID': customer_ids,
            'Age': np.random.randint(18, 70, n_customers),
            'Income': np.random.lognormal(10.5, 0.5, n_customers),
            'Education': np.random.choice(['Graduate', 'Undergraduate', 'High School', 'PhD'], n_customers),
            'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_customers),
            'Kidhome': np.random.choice([0, 1, 2, 3], n_customers, p=[0.4, 0.35, 0.2, 0.05]),
            'Teenhome': np.random.choice([0, 1, 2], n_customers, p=[0.5, 0.35, 0.15]),
            'Customer_Days': np.random.randint(1, 2000, n_customers),
        }
        
        # Generate segment-based purchase behavior
        purchase_data = []
        for i, segment in enumerate(segments):
            if segment == 'High-Value':
                wine = np.random.gamma(50, 10)
                fruits = np.random.gamma(30, 8)
                meat = np.random.gamma(80, 15)
                fish = np.random.gamma(40, 10)
                sweets = np.random.gamma(25, 5)
                gold = np.random.gamma(60, 12)
                deals = np.random.poisson(8)
                web = np.random.poisson(12)
                catalog = np.random.poisson(15)
                store = np.random.poisson(20)
                visits = np.random.poisson(10)
                
            elif segment == 'Regular':
                wine = np.random.gamma(20, 8)
                fruits = np.random.gamma(15, 5)
                meat = np.random.gamma(30, 10)
                fish = np.random.gamma(15, 6)
                sweets = np.random.gamma(10, 3)
                gold = np.random.gamma(20, 6)
                deals = np.random.poisson(4)
                web = np.random.poisson(6)
                catalog = np.random.poisson(5)
                store = np.random.poisson(8)
                visits = np.random.poisson(6)
                
            elif segment == 'Occasional':
                wine = np.random.gamma(8, 4)
                fruits = np.random.gamma(5, 3)
                meat = np.random.gamma(12, 5)
                fish = np.random.gamma(6, 3)
                sweets = np.random.gamma(4, 2)
                gold = np.random.gamma(8, 3)
                deals = np.random.poisson(2)
                web = np.random.poisson(3)
                catalog = np.random.poisson(2)
                store = np.random.poisson(4)
                visits = np.random.poisson(3)
                
            elif segment == 'New':
                wine = np.random.gamma(5, 2)
                fruits = np.random.gamma(3, 2)
                meat = np.random.gamma(8, 3)
                fish = np.random.gamma(4, 2)
                sweets = np.random.gamma(3, 1)
                gold = np.random.gamma(5, 2)
                deals = np.random.poisson(1)
                web = np.random.poisson(2)
                catalog = np.random.poisson(1)
                store = np.random.poisson(2)
                visits = np.random.poisson(2)
                
            else:  # Churned
                wine = np.random.gamma(2, 1)
                fruits = np.random.gamma(1, 1)
                meat = np.random.gamma(3, 2)
                fish = np.random.gamma(2, 1)
                sweets = np.random.gamma(1, 1)
                gold = np.random.gamma(2, 1)
                deals = np.random.poisson(0.5)
                web = np.random.poisson(1)
                catalog = np.random.poisson(0.5)
                store = np.random.poisson(1)
                visits = np.random.poisson(1)
            
            purchase_data.append([wine, fruits, meat, fish, sweets, gold, deals, web, catalog, store, visits])
        
        # Add purchase data to main data
        purchase_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                          'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                          'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
                          'NumWebVisitsMonth']
        
        for i, col in enumerate(purchase_columns):
            data[col] = [max(0, int(row[i])) for row in purchase_data]
        
        # Add campaign responses
        campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
        for col in campaign_cols:
            data[col] = np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
        
        # Add response to last campaign
        data['Response'] = np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
        
        # Add recency (days since last purchase)
        data['Recency'] = np.random.randint(0, 100, n_customers)
        
        # Add complaint flag
        data['Complain'] = np.random.choice([0, 1], n_customers, p=[0.95, 0.05])
        
        # Store true segments for validation
        data['True_Segment'] = segments
        
        self.df = pd.DataFrame(data)
        print(" Sample e-commerce dataset created successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Segments distribution: {pd.Series(segments).value_counts()}")
        
        return self.df
    
    def explore_data(self):
        """Comprehensive data exploration and analysis"""
        
        print(" DATA EXPLORATION & ANALYSIS")
       
        
        # Basic dataset information
        print("\n1.  DATASET OVERVIEW:")
        print(f"    Shape: {self.df.shape}")
        print(f"    Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"    Unique Customers: {self.df['CustomerID'].nunique()}")
        
        # Data types and missing values
        print(f"\n2.  DATA QUALITY CHECK:")
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print("   Missing values found:")
            print(missing_data[missing_data > 0])
        else:
            print("    No missing values detected!")
        
        # Basic statistics
        print(f"\n3.  DESCRIPTIVE STATISTICS:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe().round(2))
        
        # Create exploration visualizations
        self.create_exploration_visualizations()
        
        return self.df.info()
    
    def create_exploration_visualizations(self):
        """Create comprehensive data exploration visualizations"""
        # Set up the plotting area
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('Customer Data Exploration Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Age distribution
        axes[0, 0].hist(self.df['Age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Income distribution
        axes[0, 1].hist(np.log1p(self.df['Income']), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Income Distribution (Log Scale)')
        axes[0, 1].set_xlabel('Log(Income + 1)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Total spending
        spending_cols = [col for col in self.df.columns if col.startswith('Mnt')]
        self.df['Total_Spending'] = self.df[spending_cols].sum(axis=1)
        axes[0, 2].hist(self.df['Total_Spending'], bins=30, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 2].set_title('Total Spending Distribution')
        axes[0, 2].set_xlabel('Total Spending')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Customer tenure
        axes[0, 3].hist(self.df['Customer_Days'], bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[0, 3].set_title('Customer Tenure (Days)')
        axes[0, 3].set_xlabel('Days as Customer')
        axes[0, 3].set_ylabel('Frequency')
        
        # 5. Spending by category
        spending_by_category = self.df[spending_cols].mean()
        axes[1, 0].bar(range(len(spending_by_category)), spending_by_category.values, color='lightblue')
        axes[1, 0].set_title('Average Spending by Product Category')
        axes[1, 0].set_xticks(range(len(spending_by_category)))
        axes[1, 0].set_xticklabels([col.replace('Mnt', '') for col in spending_cols], rotation=45)
        axes[1, 0].set_ylabel('Average Spending')
        
        # 6. Purchase channels
        channel_cols = [col for col in self.df.columns if 'Purchases' in col]
        channel_totals = self.df[channel_cols].sum()
        axes[1, 1].pie(channel_totals.values, labels=[col.replace('Num', '').replace('Purchases', '') 
                      for col in channel_cols], autopct='%1.1f%%')
        axes[1, 1].set_title('Purchase Channel Distribution')
        
        # 7. Education vs Income
        education_income = self.df.groupby('Education')['Income'].mean().sort_values(ascending=False)
        axes[1, 2].bar(range(len(education_income)), education_income.values, color='lightcoral')
        axes[1, 2].set_title('Average Income by Education Level')
        axes[1, 2].set_xticks(range(len(education_income)))
        axes[1, 2].set_xticklabels(education_income.index, rotation=45)
        axes[1, 2].set_ylabel('Average Income')
        
        # 8. Marital Status distribution
        marital_counts = self.df['Marital_Status'].value_counts()
        axes[1, 3].pie(marital_counts.values, labels=marital_counts.index, autopct='%1.1f%%')
        axes[1, 3].set_title('Marital Status Distribution')
        
        # 9. Age vs Spending scatter plot
        axes[2, 0].scatter(self.df['Age'], self.df['Total_Spending'], alpha=0.6)
        axes[2, 0].set_title('Age vs Total Spending')
        axes[2, 0].set_xlabel('Age')
        axes[2, 0].set_ylabel('Total Spending')
        
        # 10. Income vs Spending scatter plot
        axes[2, 1].scatter(self.df['Income'], self.df['Total_Spending'], alpha=0.6)
        axes[2, 1].set_title('Income vs Total Spending')
        axes[2, 1].set_xlabel('Income')
        axes[2, 1].set_ylabel('Total Spending')
        
        # 11. Recency distribution
        axes[2, 2].hist(self.df['Recency'], bins=30, alpha=0.7, color='mediumpurple', edgecolor='black')
        axes[2, 2].set_title('Recency Distribution (Days Since Last Purchase)')
        axes[2, 2].set_xlabel('Days Since Last Purchase')
        axes[2, 2].set_ylabel('Frequency')
        
        # 12. Campaign acceptance rate
        campaign_cols = [col for col in self.df.columns if 'AcceptedCmp' in col] + ['Response']
        campaign_acceptance = self.df[campaign_cols].mean() * 100
        axes[2, 3].bar(range(len(campaign_acceptance)), campaign_acceptance.values, color='lightsteelblue')
        axes[2, 3].set_title('Campaign Acceptance Rates')
        axes[2, 3].set_xticks(range(len(campaign_acceptance)))
        axes[2, 3].set_xticklabels([col.replace('AcceptedCmp', 'Camp') for col in campaign_cols], rotation=45)
        axes[2, 3].set_ylabel('Acceptance Rate (%)')
        
        # 13. Kids at home vs Spending
        kids_spending = self.df.groupby('Kidhome')['Total_Spending'].mean()
        axes[3, 0].bar(kids_spending.index, kids_spending.values, color='lightgreen')
        axes[3, 0].set_title('Average Spending by Number of Kids at Home')
        axes[3, 0].set_xlabel('Number of Kids at Home')
        axes[3, 0].set_ylabel('Average Total Spending')
        
        # 14. Web visits vs Web purchases
        axes[3, 1].scatter(self.df['NumWebVisitsMonth'], self.df['NumWebPurchases'], alpha=0.6)
        axes[3, 1].set_title('Web Visits vs Web Purchases')
        axes[3, 1].set_xlabel('Web Visits per Month')
        axes[3, 1].set_ylabel('Web Purchases')
        
        # 15. Correlation heatmap (top correlations)
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        # Select top correlated features with Total_Spending
        spending_corr = corr_matrix['Total_Spending'].abs().sort_values(ascending=False)
        top_features = spending_corr.head(10).index
        
        sns.heatmap(corr_matrix.loc[top_features, top_features], 
                   annot=True, cmap='coolwarm', center=0, 
                   ax=axes[3, 2], fmt='.2f', square=True)
        axes[3, 2].set_title('Correlation Matrix (Top Features)')
        
        # 16. Customer lifecycle (Days vs Spending)
        axes[3, 3].scatter(self.df['Customer_Days'], self.df['Total_Spending'], alpha=0.6, color='orange')
        axes[3, 3].set_title('Customer Lifecycle: Tenure vs Spending')
        axes[3, 3].set_xlabel('Days as Customer')
        axes[3, 3].set_ylabel('Total Spending')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_rfm_metrics(self):
        """Calculate RFM (Recency, Frequency, Monetary) metrics"""
        
        print(" RFM ANALYSIS")
        
        
        # Calculate RFM metrics
        purchase_cols = [col for col in self.df.columns if 'Purchases' in col]
        spending_cols = [col for col in self.df.columns if col.startswith('Mnt')]
        
        rfm_data = pd.DataFrame()
        rfm_data['CustomerID'] = self.df['CustomerID']
        
        # Recency: Days since last purchase (lower is better)
        rfm_data['Recency'] = self.df['Recency']
        
        # Frequency: Total number of purchases
        rfm_data['Frequency'] = self.df[purchase_cols].sum(axis=1)
        
        # Monetary: Total amount spent
        rfm_data['Monetary'] = self.df[spending_cols].sum(axis=1)
        
        # Create RFM scores (1-5 scale)
        rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'], 5, labels=[1,2,3,4,5])
        rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        rfm_data['R_Score'] = rfm_data['R_Score'].astype(int)
        rfm_data['F_Score'] = rfm_data['F_Score'].astype(int)
        rfm_data['M_Score'] = rfm_data['M_Score'].astype(int)
        
        # Create RFM combined score
        rfm_data['RFM_Score'] = rfm_data['R_Score'].astype(str) + rfm_data['F_Score'].astype(str) + rfm_data['M_Score'].astype(str)
        
        # Define customer segments based on RFM scores
        def segment_customers(row):
            if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'New Customers'
            elif row['RFM_Score'] in ['512', '511', '221', '213', '231', '241']:
                return 'Promising'
            elif row['RFM_Score'] in ['155', '254', '245', '253', '252', '243', '244']:
                return 'Need Attention'
            elif row['RFM_Score'] in ['331', '321', '231', '241', '251']:
                return 'About to Sleep'
            elif row['RFM_Score'] in ['155', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['RFM_Score'] in ['222', '223', '232', '322', '231']:
                return 'Cannot Lose Them'
            elif row['RFM_Score'] in ['111', '112', '121', '131', '141', '151']:
                return 'Hibernating'
            else:
                return 'Others'
        
        rfm_data['RFM_Segment'] = rfm_data.apply(segment_customers, axis=1)
        
        self.rfm_data = rfm_data
        
        # Display results
        print("\n RFM METRICS SUMMARY:")
        print(rfm_data[['Recency', 'Frequency', 'Monetary']].describe().round(2))
        
        print(f"\n RFM SEGMENTS DISTRIBUTION:")
        segment_counts = rfm_data['RFM_Segment'].value_counts()
        for segment, count in segment_counts.items():
            percentage = (count / len(rfm_data)) * 100
            print(f"    {segment}: {count} customers ({percentage:.1f}%)")
        
        # Create RFM visualization
        self.visualize_rfm_analysis()
        
        return rfm_data
    
    def visualize_rfm_analysis(self):
        """Create RFM analysis visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RFM Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. RFM Score distribution
        axes[0, 0].hist(self.rfm_data['Recency'], bins=30, alpha=0.7, color='red', label='Recency')
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].set_xlabel('Days Since Last Purchase')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(self.rfm_data['Frequency'], bins=30, alpha=0.7, color='blue', label='Frequency')
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Number of Purchases')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[0, 2].hist(self.rfm_data['Monetary'], bins=30, alpha=0.7, color='green', label='Monetary')
        axes[0, 2].set_title('Monetary Distribution')
        axes[0, 2].set_xlabel('Total Spending')
        axes[0, 2].set_ylabel('Frequency')
        
        # 2. RFM Segments
        segment_counts = self.rfm_data['RFM_Segment'].value_counts()
        axes[1, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('RFM Segments Distribution')
        
        # 3. RFM Score heatmap
        rfm_agg = self.rfm_data.groupby('RFM_Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        
        sns.heatmap(rfm_agg.T, annot=True, cmap='RdYlBu_r', ax=axes[1, 1])
        axes[1, 1].set_title('RFM Segments Characteristics')
        
        # 4. Frequency vs Monetary scatter plot colored by Recency
        scatter = axes[1, 2].scatter(self.rfm_data['Frequency'], self.rfm_data['Monetary'], 
                                   c=self.rfm_data['Recency'], cmap='RdYlBu_r', alpha=0.6)
        axes[1, 2].set_title('Frequency vs Monetary (colored by Recency)')
        axes[1, 2].set_xlabel('Frequency')
        axes[1, 2].set_ylabel('Monetary')
        plt.colorbar(scatter, ax=axes[1, 2], label='Recency (days)')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_clustering_data(self):
        """Prepare data for clustering analysis"""
        
        print("PREPARING DATA FOR CLUSTERING")
        
        
        # Select features for clustering
        clustering_features = []
        
        # Demographic features
        clustering_features.extend(['Age', 'Income', 'Customer_Days'])
        
        # Spending features
        spending_cols = [col for col in self.df.columns if col.startswith('Mnt')]
        clustering_features.extend(spending_cols)
        
        # Purchase behavior features
        purchase_cols = [col for col in self.df.columns if 'Purchases' in col]
        clustering_features.extend(purchase_cols)
        
        # Additional behavioral features
        clustering_features.extend(['NumWebVisitsMonth', 'Recency'])
        
        # Create feature matrix
        self.X = self.df[clustering_features].copy()
        
        # Handle any missing values
        self.X = self.X.fillna(self.X.median())
        
        # Add derived features
        self.X['Total_Spending'] = self.df[spending_cols].sum(axis=1)
        self.X['Total_Purchases'] = self.df[purchase_cols].sum(axis=1)
        self.X['Avg_Order_Value'] = self.X['Total_Spending'] / (self.X['Total_Purchases'] + 1)
        self.X['Spending_per_Day'] = self.X['Total_Spending'] / (self.X['Customer_Days'] + 1)
        
        # Scale the features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f" Clustering dataset prepared!")
        print(f"    Features: {self.X.shape[1]}")
        print(f"    Samples: {self.X.shape[0]}")
        print(f"    Feature names: {list(self.X.columns)}")
        
        return self.X, self.X_scaled
    
    def find_optimal_clusters(self, max_clusters=12):
        """Find optimal number of clusters using multiple methods"""
        
        print(" FINDING OPTIMAL NUMBER OF CLUSTERS")
        
        
        # Prepare data
        self.prepare_clustering_data()
        
        # Initialize lists to store metrics
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        k_range = range(2, max_clusters + 1)
        
        print("Testing different numbers of clusters...")
        
        for k in k_range:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X_scaled)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(self.X_scaled, cluster_labels))
            
            print(f"   k={k}: Silhouette={silhouette_scores[-1]:.3f}, Calinski-Harabasz={calinski_scores[-1]:.1f}")
        
        # Find optimal k using elbow method and silhouette score
        # Elbow method: find the point of maximum curvature
        elbow_k = self.find_elbow_point(k_range, inertias)
        
        # Best silhouette score
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        # Best Calinski-Harabasz score
        best_calinski_k = k_range[np.argmax(calinski_scores)]
        
        print(f"\n OPTIMAL CLUSTER RECOMMENDATIONS:")
        print(f"    Elbow Method: {elbow_k} clusters")
        print(f"    Best Silhouette Score: {best_silhouette_k} clusters (score: {max(silhouette_scores):.3f})")
        print(f"    Best Calinski-Harabasz: {best_calinski_k} clusters (score: {max(calinski_scores):.1f})")
        
        # Choose optimal k (prioritize silhouette score)
        self.optimal_clusters = best_silhouette_k
        print(f"\n Selected optimal clusters: {self.optimal_clusters}")
        
        # Visualize cluster evaluation metrics
        self.plot_cluster_metrics(k_range, inertias, silhouette_scores, calinski_scores)
        
        return self.optimal_clusters
    
    def find_elbow_point(self, k_range, inertias):
        """Find elbow point using the knee locator method"""
        # Calculate the differences
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        
        # Find the point with maximum second derivative (most curvature)
        if len(diffs2) > 0:
            elbow_idx = np.argmax(diffs2) + 2  # +2 because of double diff
            return k_range[min(elbow_idx, len(k_range) - 1)]
        else:
            return k_range[len(k_range) // 2]  # Default to middle if calculation fails
    
    def plot_cluster_metrics(self, k_range, inertias, silhouette_scores, calinski_scores):
        """Plot cluster evaluation metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Elbow plot
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_title('Elbow Method for Optimal K')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia (Within-cluster Sum of Squares)')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette score plot
        axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[1].set_title('Silhouette Score vs Number of Clusters')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz score plot
        axes[2].plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
        axes[2].set_title('Calinski-Harabasz Score vs Number of Clusters')
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Calinski-Harabasz Score')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def perform_clustering(self, n_clusters=None):
        """Perform customer segmentation using K-means clustering"""
        
        print(" PERFORMING CUSTOMER SEGMENTATION")
       
        if n_clusters is None:
            if self.optimal_clusters is None:
                self.find_optimal_clusters()
            n_clusters = self.optimal_clusters
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['Cluster'] = self.kmeans_model.fit_predict(self.X_scaled)
        
        # Calculate cluster performance metrics
        silhouette_avg = silhouette_score(self.X_scaled, self.df['Cluster'])
        calinski_score = calinski_harabasz_score(self.X_scaled, self.df['Cluster'])
        
        print(f" K-means clustering completed with {n_clusters} clusters")
        print(f"    Silhouette Score: {silhouette_avg:.3f}")
        print(f"    Calinski-Harabasz Score: {calinski_score:.1f}")
        
        # Analyze clusters
        self.analyze_clusters()
        
        # Create cluster visualizations
        self.visualize_clusters()
        
        return self.df['Cluster']
    
    def analyze_clusters(self):
        """Analyze and interpret customer clusters"""
        
        print(" CLUSTER ANALYSIS & INSIGHTS")
        
        
        # Cluster size distribution
        cluster_sizes = self.df['Cluster'].value_counts().sort_index()
        print(f"\n CLUSTER SIZES:")
        for cluster_id, size in cluster_sizes.items():
            percentage = (size / len(self.df)) * 100
            print(f"    Cluster {cluster_id}: {size} customers ({percentage:.1f}%)")
        
        # Analyze cluster characteristics
        cluster_analysis = []
        
        for cluster_id in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            
            analysis = {
                'Cluster': cluster_id,
                'Size': len(cluster_data),
                'Avg_Age': cluster_data['Age'].mean(),
                'Avg_Income': cluster_data['Income'].mean(),
                'Avg_Total_Spending': cluster_data[[col for col in self.df.columns if col.startswith('Mnt')]].sum(axis=1).mean(),
                'Avg_Total_Purchases': cluster_data[[col for col in self.df.columns if 'Purchases' in col]].sum(axis=1).mean(),
                'Avg_Customer_Days': cluster_data['Customer_Days'].mean(),
                'Avg_Recency': cluster_data['Recency'].mean(),
                'Web_Visit_Rate': cluster_data['NumWebVisitsMonth'].mean(),
                'Campaign_Response_Rate': cluster_data[[col for col in self.df.columns if col.startswith('AcceptedCmp')]].mean().mean() * 100
            }
            
            cluster_analysis.append(analysis)
        
        # Create cluster summary DataFrame
        self.cluster_summary = pd.DataFrame(cluster_analysis)
        
        print(f"\n CLUSTER CHARACTERISTICS:")
        print(self.cluster_summary.round(2))
        
        # Generate cluster insights and names
        self.generate_cluster_insights()
        
        return self.cluster_summary
    
    def generate_cluster_insights(self):
        """Generate insights and meaningful names for each cluster"""
        print(f"\n CLUSTER INSIGHTS & SEGMENTATION:")
        
        cluster_names = {}
        cluster_insights = {}
        
        for _, row in self.cluster_summary.iterrows():
            cluster_id = int(row['Cluster'])
            
            # Determine cluster characteristics
            high_income = row['Avg_Income'] > self.df['Income'].median()
            high_spending = row['Avg_Total_Spending'] > self.df[[col for col in self.df.columns if col.startswith('Mnt')]].sum(axis=1).median()
            high_frequency = row['Avg_Total_Purchases'] > self.df[[col for col in self.df.columns if 'Purchases' in col]].sum(axis=1).median()
            recent_activity = row['Avg_Recency'] < self.df['Recency'].median()
            high_engagement = row['Web_Visit_Rate'] > self.df['NumWebVisitsMonth'].median()
            
            # Generate cluster name and insights
            if high_income and high_spending and high_frequency:
                name = " VIP Champions"
                insight = "High-value customers with premium spending patterns and frequent purchases"
            elif high_spending and recent_activity:
                name = " Loyal Customers"
                insight = "Regular spenders with consistent recent activity"
            elif high_frequency and high_engagement:
                name = " Active Shoppers"
                insight = "Frequent buyers who actively engage with the platform"
            elif recent_activity and not high_spending:
                name = " New Customers"
                insight = "Recent customers with potential for growth"
            elif not recent_activity and high_spending:
                name = " At-Risk High Value"
                insight = "Previously high-value customers showing signs of churn"
            elif not recent_activity and not high_spending:
                name = " Hibernating"
                insight = "Inactive customers requiring re-engagement campaigns"
            else:
                name = f" Segment {cluster_id}"
                insight = "Mixed characteristics requiring detailed analysis"
            
            cluster_names[cluster_id] = name
            cluster_insights[cluster_id] = insight
            
            print(f"\n   {name}")
            print(f"    Size: {row['Size']} customers ({row['Size']/len(self.df)*100:.1f}%)")
            print(f"    Average Income: ${row['Avg_Income']:,.0f}")
            print(f"    Average Spending: ${row['Avg_Total_Spending']:,.0f}")
            print(f"    Average Purchases: {row['Avg_Total_Purchases']:.1f}")
            print(f"    Days Since Last Purchase: {row['Avg_Recency']:.0f}")
            print(f"    Insight: {insight}")
        
        self.cluster_names = cluster_names
        self.cluster_insights = cluster_insights
        
        return cluster_names, cluster_insights
    
    def visualize_clusters(self):
        """Create comprehensive cluster visualizations"""
        print("\n Creating cluster visualizations...")
        
        # Create multiple visualization plots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. PCA visualization (2D)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        plt.subplot(3, 3, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.df['Cluster'], cmap='tab10', alpha=0.7)
        plt.title('Customer Clusters (PCA Visualization)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, label='Cluster')
        
        # 2. Income vs Total Spending
        spending_cols = [col for col in self.df.columns if col.startswith('Mnt')]
        total_spending = self.df[spending_cols].sum(axis=1)
        
        plt.subplot(3, 3, 2)
        scatter = plt.scatter(self.df['Income'], total_spending, c=self.df['Cluster'], cmap='tab10', alpha=0.7)
        plt.title('Income vs Total Spending by Cluster')
        plt.xlabel('Income')
        plt.ylabel('Total Spending')
        plt.colorbar(scatter, label='Cluster')
        
        # 3. Age vs Recency
        plt.subplot(3, 3, 3)
        scatter = plt.scatter(self.df['Age'], self.df['Recency'], c=self.df['Cluster'], cmap='tab10', alpha=0.7)
        plt.title('Age vs Recency by Cluster')
        plt.xlabel('Age')
        plt.ylabel('Days Since Last Purchase')
        plt.colorbar(scatter, label='Cluster')
        
        # 4. Cluster size distribution
        plt.subplot(3, 3, 4)
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        bars = plt.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.tab10(np.arange(len(cluster_counts))))
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Customers')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5, f'{height}', 
                    ha='center', va='bottom')
        
        # 5. Average spending by cluster
        plt.subplot(3, 3, 5)
        # Calculate total spending per customer
        total_spending_per_customer = self.df[spending_cols].sum(axis=1)
        avg_spending = self.df.groupby('Cluster').apply(lambda x: x[spending_cols].sum(axis=1).mean())
        bars = plt.bar(avg_spending.index, avg_spending.values, color=plt.cm.tab10(np.arange(len(avg_spending))))
        plt.title('Average Total Spending by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Average Total Spending')
        plt.xticks(avg_spending.index)
        
        # 6. Purchase frequency by cluster
        purchase_cols = [col for col in self.df.columns if 'Purchases' in col]
        plt.subplot(3, 3, 6)
        avg_purchases = self.df.groupby('Cluster').apply(lambda x: x[purchase_cols].sum(axis=1).mean())
        bars = plt.bar(avg_purchases.index, avg_purchases.values, color=plt.cm.tab10(np.arange(len(avg_purchases))))
        plt.title('Average Total Purchases by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Average Total Purchases')
        plt.xticks(avg_purchases.index)
        
        # 7. Customer tenure by cluster
        plt.subplot(3, 3, 7)
        avg_tenure = self.df.groupby('Cluster')['Customer_Days'].mean()
        bars = plt.bar(avg_tenure.index, avg_tenure.values, color=plt.cm.tab10(np.arange(len(avg_tenure))))
        plt.title('Average Customer Tenure by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Average Days as Customer')
        plt.xticks(avg_tenure.index)
        
        # 8. Spending pattern heatmap
        plt.subplot(3, 3, 8)
        cluster_spending = self.df.groupby('Cluster')[spending_cols].mean()
        sns.heatmap(cluster_spending.T, annot=True, cmap='YlOrRd', fmt='.0f')
        plt.title('Average Spending Patterns by Cluster')
        plt.ylabel('Product Categories')
        
        # 9. 3D scatter plot (Income, Spending, Age)
        ax = fig.add_subplot(3, 3, 9, projection='3d')
        colors = plt.cm.tab10(self.df['Cluster'])
        ax.scatter(self.df['Income'], total_spending, self.df['Age'], c=colors, alpha=0.6)
        ax.set_xlabel('Income')
        ax.set_ylabel('Total Spending')
        ax.set_zlabel('Age')
        ax.set_title('3D Cluster Visualization')
        
        plt.tight_layout()
        plt.show()
    
    def compare_clustering_algorithms(self):
        """Compare different clustering algorithms"""
        
        print(" COMPARING CLUSTERING ALGORITHMS")
       
        # Prepare data
        if not hasattr(self, 'X_scaled'):
            self.prepare_clustering_data()
        
        # Define algorithms to compare
        algorithms = {
            'K-Means': KMeans(n_clusters=5, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Agglomerative': AgglomerativeClustering(n_clusters=5)
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            print(f"\nTesting {name}...")
            
            # Fit the algorithm
            if name == 'DBSCAN':
                labels = algorithm.fit_predict(self.X_scaled)
            else:
                labels = algorithm.fit_predict(self.X_scaled)
            
            # Calculate metrics (skip if only one cluster or noise points)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > 1:
                silhouette = silhouette_score(self.X_scaled, labels)
                if n_clusters > 1 and len(set(labels)) > 1:
                    calinski = calinski_harabasz_score(self.X_scaled, labels)
                else:
                    calinski = 0
            else:
                silhouette = -1
                calinski = 0
            
            results[name] = {
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'calinski_score': calinski,
                'labels': labels
            }
            
            print(f"    Clusters found: {n_clusters}")
            print(f"    Silhouette Score: {silhouette:.3f}")
            print(f"    Calinski-Harabasz Score: {calinski:.1f}")
        
        # Display comparison
        comparison_df = pd.DataFrame({
            name: [info['n_clusters'], info['silhouette_score'], info['calinski_score']]
            for name, info in results.items()
        }, index=['Number of Clusters', 'Silhouette Score', 'Calinski-Harabasz Score'])
        
        print(f"\n ALGORITHM COMPARISON:")
        print(comparison_df.round(3))
        
        return results
    
    def generate_business_recommendations(self):
        """Generate actionable business recommendations based on segmentation"""
        
        print(" BUSINESS RECOMMENDATIONS & STRATEGIES")
       
        
        recommendations = {}
        
        for cluster_id, name in self.cluster_names.items():
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            size = len(cluster_data)
            
            # Calculate key metrics
            avg_spending = cluster_data[[col for col in self.df.columns if col.startswith('Mnt')]].sum(axis=1).mean()
            avg_purchases = cluster_data[[col for col in self.df.columns if 'Purchases' in col]].sum(axis=1).mean()
            avg_recency = cluster_data['Recency'].mean()
            campaign_response = cluster_data[[col for col in self.df.columns if col.startswith('AcceptedCmp')]].mean().mean()
            
            # Generate specific recommendations
            recs = []
            
            if 'VIP' in name or 'Champions' in name:
                recs.extend([
                    " Implement VIP loyalty program with exclusive perks",
                    " Send personalized premium product recommendations",
                    " Offer early access to new products and sales",
                    " Provide dedicated customer service line",
                    " Create premium membership tiers with special benefits"
                ])
            
            elif 'Loyal' in name:
                recs.extend([
                    " Reward loyalty with points-based system",
                    " Send regular engagement emails with relevant offers",
                    " Celebrate customer milestones and anniversaries",
                    " Cross-sell complementary products",
                    " Monitor for any signs of decreased activity"
                ])
            
            elif 'Active' in name or 'Shoppers' in name:
                recs.extend([
                    " Optimize website experience and mobile app",
                    " Implement real-time personalized recommendations",
                    " Offer subscription services for frequently bought items",
                    " Use retargeting ads for cart abandonment",
                    " Introduce product bundles and bulk discounts"
                ])
            
            elif 'New' in name:
                recs.extend([
                    " Design comprehensive onboarding experience",
                    " Offer welcome discounts and first-purchase incentives",
                    " Provide educational content about products",
                    " Send regular check-in emails with tips and offers",
                    " Implement referral program to expand network"
                ])
            
            elif 'At-Risk' in name:
                recs.extend([
                    " Implement win-back campaigns with special offers",
                    " Conduct surveys to understand satisfaction issues",
                    " Offer significant discounts to re-engage",
                    " Send 'We miss you' personalized messages",
                    " Test different communication channels"
                ])
            
            elif 'Hibernating' in name:
                recs.extend([
                    " Create reactivation campaign series",
                    " Use different messaging and channels",
                    " Offer steep discounts or free shipping",
                    " A/B test different re-engagement strategies",
                    " Consider moving to less frequent communication"
                ])
            
            recommendations[cluster_id] = {
                'name': name,
                'size': size,
                'percentage': (size / len(self.df)) * 100,
                'avg_spending': avg_spending,
                'recommendations': recs
            }
        
        # Display recommendations
        for cluster_id, info in recommendations.items():
            print(f"\n{info['name']}")
            print(f" Size: {info['size']} customers ({info['percentage']:.1f}%)")
            print(f" Avg. Spending: ${info['avg_spending']:,.0f}")
            print(f" Recommended Strategies:")
            for rec in info['recommendations']:
                print(f"    {rec}")
        
        # Overall business strategy
        print(" OVERALL BUSINESS STRATEGY")
        
        
        total_customers = len(self.df)
        high_value_customers = len(self.df[self.df['Cluster'].isin([cluster_id for cluster_id, name in self.cluster_names.items() if 'VIP' in name or 'Champions' in name])])
        at_risk_customers = len(self.df[self.df['Cluster'].isin([cluster_id for cluster_id, name in self.cluster_names.items() if 'At-Risk' in name])])
        
        print(f" GROWTH OPPORTUNITIES:")
        print(f"    Focus on converting {len(self.df[self.df['Cluster'].isin([cluster_id for cluster_id, name in self.cluster_names.items() if 'New' in name])])} new customers to loyal customers")
        print(f"    Retain {high_value_customers} high-value customers with premium experiences")
        print(f"    Re-engage {at_risk_customers} at-risk customers before they churn")
        
        print(f"\n MARKETING BUDGET ALLOCATION:")
        print(f"    40% - Customer retention (loyal and high-value segments)")
        print(f"    30% - Customer acquisition (new customer conversion)")
        print(f"    20% - Win-back campaigns (at-risk and hibernating)")
        print(f"    10% - Testing and optimization")
        
        return recommendations
    
    def run_complete_analysis(self, file_path=None):
        """Run the complete customer segmentation analysis"""
        print(" CUSTOMER SEGMENTATION ANALYSIS FOR E-COMMERCE")
       
        
        try:
            # 1. Load and explore data
            print("\n Step 1: Loading and exploring data...")
            self.load_data(file_path)
            self.explore_data()
            
            # 2. Calculate RFM metrics
            print("\n Step 2: Calculating RFM metrics...")
            self.calculate_rfm_metrics()
            
            # 3. Find optimal clusters
            print("\n Step 3: Finding optimal number of clusters...")
            self.find_optimal_clusters()
            
            # 4. Perform clustering
            print("\n Step 4: Performing customer segmentation...")
            self.perform_clustering()
            
            # 5. Compare algorithms (optional)
            print("\n Step 5: Comparing clustering algorithms...")
            self.compare_clustering_algorithms()
            
            # 6. Generate business recommendations
            print("\n Step 6: Generating business recommendations...")
            recommendations = self.generate_business_recommendations()
            
            
            print(" CUSTOMER SEGMENTATION ANALYSIS COMPLETE!")
           
            print(" Key Deliverables:")
            print("    Customer segments identified and characterized")
            print("    RFM analysis for customer value assessment")
            print("    Actionable business recommendations")
            print("    Comprehensive visualizations and insights")
            
            return {
                'clusters': self.df['Cluster'],
                'cluster_summary': self.cluster_summary,
                'recommendations': recommendations,
                'rfm_data': self.rfm_data
            }
            
        except Exception as e:
            print(f" Error in analysis: {str(e)}")
            print("Please check your data and try again.")
            return None

# Example usage and main execution
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = CustomerSegmentationAnalysis()
    
    print(" Starting Customer Segmentation Analysis...")
    print("Choose an option:")
    print("1. Use sample data (recommended for demonstration)")
    print("2. Load your own dataset")
    
    # For demonstration, we'll use sample data
    print("\n Running analysis with sample data...")
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    if results:
        
        print(" ANALYSIS SUMMARY")
       
        
        # Display key metrics
        n_clusters = len(results['cluster_summary'])
        print(f" Successfully identified {n_clusters} customer segments")
        
        # Show top insights
        print(f"\n TOP INSIGHTS:")
        if hasattr(analyzer, 'cluster_names'):
            for cluster_id, name in analyzer.cluster_names.items():
                size = len(analyzer.df[analyzer.df['Cluster'] == cluster_id])
                percentage = (size / len(analyzer.df)) * 100
                print(f"    {name}: {size} customers ({percentage:.1f}%)")
        
        print(f"\n NEXT STEPS:")
        print("   1. Implement recommended strategies for each segment")
        print("   2. Set up monitoring dashboards for segment performance")
        print("   3. Test different approaches with A/B experiments")
        print("   4. Regular re-segmentation (quarterly recommended)")
        print("   5. Integrate insights into marketing automation tools")