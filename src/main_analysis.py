import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import json

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

class EcommerceConversionAnalyzer:
    """
    Main class for e-commerce conversion analysis.
    Handles data loading, cleaning, feature engineering, and analysis.
    """
    
    def __init__(self, filepath, sample_size=None):
        """
        Initialize analyzer with data path.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV/Parquet file
        sample_size : int, optional
            Number of rows to sample for faster processing
        """
        self.filepath = filepath
        self.sample_size = sample_size
        self.df = None
        self.df_sessions = None
        self.conversion_metrics = {}
        self.insights = []
        
    def load_data(self):
        """Load and perform initial data inspection."""
        print("=" * 80)
        print("STEP 1: DATA LOADING")
        print("=" * 80)
        
        # Detect file type and load accordingly
        if self.filepath.endswith('.parquet'):
            self.df = pd.read_parquet(self.filepath)
        else:
            # Load in chunks for large CSV files
            if self.sample_size:
                self.df = pd.read_csv(self.filepath, nrows=self.sample_size)
            else:
                self.df = pd.read_csv(self.filepath)
        
        print(f"\n✓ Loaded {len(self.df):,} records")
        print(f"✓ Shape: {self.df.shape}")
        print(f"✓ Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"\n{self.df.head()}")
        
        return self
    
    def understand_data(self):
        """Perform comprehensive data understanding."""
        print("\n" + "=" * 80)
        print("STEP 2: DATA UNDERSTANDING")
        print("=" * 80)
        
        # Basic info
        print("\nDataset Overview:")
        print(self.df.info())
        
        # Missing values
        print("\nMissing Values Analysis:")
        missing = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum().values / len(self.df) * 100).round(2)
        })
        missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        print(missing)
        
        # Unique values
        print("\nUnique Values per Column:")
        for col in self.df.columns:
            print(f"{col}: {self.df[col].nunique():,} unique values")
        
        # Data types
        print("\nData Types:")
        print(self.df.dtypes)
        
        # Statistical summary
        print("\nStatistical Summary (Numeric Columns):")
        print(self.df.describe())
        
        return self
    
    def clean_and_prepare(self):
        """Clean data and create derived features."""
        print("\n" + "=" * 80)
        print("STEP 3: DATA CLEANING & FEATURE ENGINEERING")
        print("=" * 80)
        
        # Make a copy
        df = self.df.copy()
        
        # Convert timestamp to datetime
        print("\nConverting timestamps...")
        df['event_time'] = pd.to_datetime(df['event_time'])
        
        # Extract temporal features
        df['hour'] = df['event_time'].dt.hour
        df['day_of_week'] = df['event_time'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_name'] = df['event_time'].dt.day_name()
        df['date'] = df['event_time'].dt.date
        df['week'] = df['event_time'].dt.isocalendar().week
        df['month'] = df['event_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                     bins=[0, 6, 12, 18, 24],
                                     labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                     include_lowest=True)
        
        # Handle missing values
        print("\nHandling missing values...")
        
        # Fill or drop based on column
        if 'category_code' in df.columns:
            df['category_code'] = df['category_code'].fillna('unknown')
            # Extract main category
            df['main_category'] = df['category_code'].apply(
                lambda x: x.split('.')[0] if isinstance(x, str) and x != 'unknown' else 'unknown'
            )
        
        if 'brand' in df.columns:
            df['brand'] = df['brand'].fillna('unknown')
        
        # Create user session identifier (simplified: user_id + date)
        df['session_id'] = df['user_id'].astype(str) + '_' + df['date'].astype(str)
        
        # Create conversion label
        df['is_purchase'] = (df['event_type'] == 'purchase').astype(int)
        df['is_cart'] = (df['event_type'] == 'cart').astype(int)
        df['is_view'] = (df['event_type'] == 'view').astype(int)
        
        # Sort by user and time
        df = df.sort_values(['user_id', 'event_time'])
        
        print(f"✓ Created {len([c for c in df.columns if c not in self.df.columns])} new features")
        print(f"✓ New columns: {[c for c in df.columns if c not in self.df.columns]}")
        
        self.df = df
        return self
    
    def create_session_features(self):
        """Aggregate data at session level."""
        print("\n" + "=" * 80)
        print("STEP 4: SESSION-LEVEL FEATURE CREATION")
        print("=" * 80)
        
        print("\nAggregating sessions...")
        
        # Session-level aggregation
        session_agg = self.df.groupby('session_id').agg({
            'event_time': ['min', 'max', 'count'],
            'product_id': 'nunique',
            'is_purchase': 'sum',
            'is_cart': 'sum',
            'is_view': 'sum',
            'user_id': 'first',
            'hour': 'first',
            'day_of_week': 'first',
            'day_name': 'first',
            'is_weekend': 'first',
            'main_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        }).reset_index()
        
        # Flatten column names
        session_agg.columns = ['_'.join(col).strip('_') for col in session_agg.columns.values]
        session_agg.columns = ['session_id', 'session_start', 'session_end', 'total_events',
                                'unique_products_viewed', 'purchases', 'carts', 'views',
                                'user_id', 'hour', 'day_of_week', 'day_name', 'is_weekend',
                                'primary_category']
        
        # Calculate session duration in seconds
        session_agg['session_duration_sec'] = (
            session_agg['session_end'] - session_agg['session_start']
        ).dt.total_seconds()
        
        # Create conversion label
        session_agg['converted'] = (session_agg['purchases'] > 0).astype(int)
        
        # Cart abandonment
        session_agg['cart_abandonment'] = (
            (session_agg['carts'] > 0) & (session_agg['purchases'] == 0)
        ).astype(int)
        
        # User return analysis: count sessions per user
        user_session_counts = session_agg.groupby('user_id').size().reset_index(name='user_session_count')
        session_agg = session_agg.merge(user_session_counts, on='user_id', how='left')
        session_agg['is_returning_user'] = (session_agg['user_session_count'] > 1).astype(int)
        
        print(f"✓ Created {len(session_agg):,} sessions from {self.df['user_id'].nunique():,} users")
        print(f"✓ Average {len(session_agg) / self.df['user_id'].nunique():.2f} sessions per user")
        
        self.df_sessions = session_agg
        return self
    
    def calculate_conversion_metrics(self):
        """Calculate key conversion metrics."""
        print("\n" + "=" * 80)
        print("STEP 5: CONVERSION METRICS CALCULATION")
        print("=" * 80)
        
        metrics = {}
        
        # Overall metrics
        metrics['total_sessions'] = len(self.df_sessions)
        metrics['total_conversions'] = self.df_sessions['converted'].sum()
        metrics['overall_conversion_rate'] = (
            metrics['total_conversions'] / metrics['total_sessions'] * 100
        )
        
        # Event-level metrics
        metrics['total_events'] = len(self.df)
        metrics['total_views'] = self.df['is_view'].sum()
        metrics['total_carts'] = self.df['is_cart'].sum()
        metrics['total_purchases'] = self.df['is_purchase'].sum()
        
        # Funnel metrics
        metrics['view_to_cart_rate'] = (metrics['total_carts'] / metrics['total_views'] * 100)
        metrics['cart_to_purchase_rate'] = (metrics['total_purchases'] / metrics['total_carts'] * 100)
        metrics['view_to_purchase_rate'] = (metrics['total_purchases'] / metrics['total_views'] * 100)
        
        # Cart abandonment
        metrics['cart_abandonment_count'] = self.df_sessions['cart_abandonment'].sum()
        sessions_with_cart = (self.df_sessions['carts'] > 0).sum()
        metrics['cart_abandonment_rate'] = (
            metrics['cart_abandonment_count'] / sessions_with_cart * 100 if sessions_with_cart > 0 else 0
        )
        
        # User metrics
        metrics['total_users'] = self.df['user_id'].nunique()
        metrics['returning_users'] = self.df_sessions[self.df_sessions['is_returning_user'] == 1]['user_id'].nunique()
        metrics['new_users'] = metrics['total_users'] - metrics['returning_users']
        metrics['returning_user_rate'] = (metrics['returning_users'] / metrics['total_users'] * 100)
        
        print("\nKEY CONVERSION METRICS:")
        print("-" * 80)
        print(f"Total Sessions: {metrics['total_sessions']:,}")
        print(f"Total Conversions: {metrics['total_conversions']:,}")
        print(f"Overall Conversion Rate: {metrics['overall_conversion_rate']:.2f}%")
        print(f"\nFunnel Metrics:")
        print(f"  Views → Carts: {metrics['view_to_cart_rate']:.2f}%")
        print(f"  Carts → Purchases: {metrics['cart_to_purchase_rate']:.2f}%")
        print(f"  Views → Purchases: {metrics['view_to_purchase_rate']:.2f}%")
        print(f"\nCart Abandonment Rate: {metrics['cart_abandonment_rate']:.2f}%")
        print(f"Returning User Rate: {metrics['returning_user_rate']:.2f}%")
        
        self.conversion_metrics = metrics
        return self
    
    def analyze_temporal_patterns(self):
        """Analyze conversion patterns by time."""
        print("\n" + "=" * 80)
        print("STEP 6: TEMPORAL PATTERN ANALYSIS")
        print("=" * 80)
        
        # Conversion by hour
        hourly = self.df_sessions.groupby('hour').agg({
            'converted': ['sum', 'mean', 'count']
        }).reset_index()
        hourly.columns = ['hour', 'conversions', 'conversion_rate', 'sessions']
        hourly['conversion_rate'] *= 100
        hourly = hourly.sort_values('conversion_rate', ascending=False)
        
        print("\nTop 5 Hours by Conversion Rate:")
        print(hourly.head())
        
        # Conversion by day of week
        daily = self.df_sessions.groupby(['day_of_week', 'day_name']).agg({
            'converted': ['sum', 'mean', 'count']
        }).reset_index()
        daily.columns = ['day_of_week', 'day_name', 'conversions', 'conversion_rate', 'sessions']
        daily['conversion_rate'] *= 100
        daily = daily.sort_values('conversion_rate', ascending=False)
        
        print("\nConversion by Day of Week:")
        print(daily)
        
        # Weekend vs Weekday
        weekend_analysis = self.df_sessions.groupby('is_weekend').agg({
            'converted': ['sum', 'mean', 'count']
        }).reset_index()
        weekend_analysis.columns = ['is_weekend', 'conversions', 'conversion_rate', 'sessions']
        weekend_analysis['conversion_rate'] *= 100
        weekend_analysis['period'] = weekend_analysis['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
        
        print("\nWeekend vs Weekday:")
        print(weekend_analysis)
        
        # Statistical test: weekend vs weekday
        weekday_conv = self.df_sessions[self.df_sessions['is_weekend'] == 0]['converted']
        weekend_conv = self.df_sessions[self.df_sessions['is_weekend'] == 1]['converted']
        t_stat, p_val = ttest_ind(weekday_conv, weekend_conv)
        
        print(f"\nT-test (Weekend vs Weekday): t={t_stat:.4f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("✓ Statistically significant difference!")
        
        return self
    
    def analyze_behavioral_patterns(self):
        """Analyze user behavior patterns."""
        print("\n" + "=" * 80)
        print("STEP 7: BEHAVIORAL PATTERN ANALYSIS")
        print("=" * 80)
        
        # Returning vs New users
        user_type_analysis = self.df_sessions.groupby('is_returning_user').agg({
            'converted': ['sum', 'mean', 'count'],
            'session_duration_sec': 'mean',
            'unique_products_viewed': 'mean'
        }).reset_index()
        user_type_analysis.columns = ['is_returning', 'conversions', 'conversion_rate', 'sessions',
                                       'avg_duration', 'avg_products_viewed']
        user_type_analysis['conversion_rate'] *= 100
        user_type_analysis['user_type'] = user_type_analysis['is_returning'].map({0: 'New', 1: 'Returning'})
        
        print("\nReturning vs New Users:")
        print(user_type_analysis)
        
        # Session duration analysis
        converters = self.df_sessions[self.df_sessions['converted'] == 1]['session_duration_sec']
        non_converters = self.df_sessions[self.df_sessions['converted'] == 0]['session_duration_sec']
        
        print(f"\nSession Duration Analysis:")
        print(f"Converters: {converters.mean():.2f}s (median: {converters.median():.2f}s)")
        print(f"Non-converters: {non_converters.mean():.2f}s (median: {non_converters.median():.2f}s)")
        
        t_stat, p_val = ttest_ind(converters, non_converters)
        print(f"T-test: t={t_stat:.4f}, p={p_val:.4f}")
        
        # Products viewed analysis
        products_conv = self.df_sessions[self.df_sessions['converted'] == 1]['unique_products_viewed']
        products_non_conv = self.df_sessions[self.df_sessions['converted'] == 0]['unique_products_viewed']
        
        print(f"\nProducts Viewed Analysis:")
        print(f"Converters: {products_conv.mean():.2f} products (median: {products_conv.median():.2f})")
        print(f"Non-converters: {products_non_conv.mean():.2f} products (median: {products_non_conv.median():.2f})")
        
        return self
    
    def analyze_category_patterns(self):
        """Analyze conversion by product category."""
        print("\n" + "=" * 80)
        print("STEP 8: CATEGORY PATTERN ANALYSIS")
        print("=" * 80)
        
        # Conversion by category
        category_analysis = self.df_sessions.groupby('primary_category').agg({
            'converted': ['sum', 'mean', 'count']
        }).reset_index()
        category_analysis.columns = ['category', 'conversions', 'conversion_rate', 'sessions']
        category_analysis['conversion_rate'] *= 100
        category_analysis = category_analysis[category_analysis['sessions'] >= 100]  # Filter low volume
        category_analysis = category_analysis.sort_values('conversion_rate', ascending=False)
        
        print("\nTop 10 Categories by Conversion Rate:")
        print(category_analysis.head(10))
        
        # Chi-square test: category independence
        if 'main_category' in self.df.columns:
            contingency_table = pd.crosstab(
                self.df_sessions['primary_category'],
                self.df_sessions['converted']
            )
            chi2, p_val, dof, expected = chi2_contingency(contingency_table)
            print(f"\nChi-square test (Category vs Conversion):")
            print(f"χ²={chi2:.4f}, p={p_val:.4f}, df={dof}")
            if p_val < 0.05:
                print("✓ Category significantly influences conversion!")
        
        return self
    
    def generate_insights(self):
        """Generate key insights from analysis."""
        print("\n" + "=" * 80)
        print("STEP 9: KEY INSIGHTS GENERATION")
        print("=" * 80)
        
        insights = []
        
        # Overall conversion
        insights.append(
            f"Overall conversion rate is {self.conversion_metrics['overall_conversion_rate']:.2f}%, "
            f"with {self.conversion_metrics['total_conversions']:,} conversions from "
            f"{self.conversion_metrics['total_sessions']:,} sessions."
        )
        
        # Funnel insights
        insights.append(
            f"Conversion funnel: {self.conversion_metrics['view_to_cart_rate']:.2f}% of views add to cart, "
            f"and {self.conversion_metrics['cart_to_purchase_rate']:.2f}% of carts complete purchase."
        )
        
        # Cart abandonment
        insights.append(
            f"Cart abandonment rate is {self.conversion_metrics['cart_abandonment_rate']:.2f}%, "
            f"representing a significant opportunity for recovery campaigns."
        )
        
        # Temporal patterns
        hourly_best = self.df_sessions.groupby('hour')['converted'].mean().idxmax()
        hourly_best_rate = self.df_sessions.groupby('hour')['converted'].mean().max() * 100
        insights.append(
            f"Peak conversion hour is {hourly_best}:00 with {hourly_best_rate:.2f}% conversion rate, "
            f"suggesting optimal timing for promotions."
        )
        
        # Day patterns
        daily_best = self.df_sessions.groupby('day_name')['converted'].mean().idxmax()
        daily_best_rate = self.df_sessions.groupby('day_name')['converted'].mean().max() * 100
        insights.append(
            f"{daily_best} has the highest conversion rate at {daily_best_rate:.2f}%, "
            f"indicating day-of-week effects on purchasing behavior."
        )
        
        # User type
        new_rate = self.df_sessions[self.df_sessions['is_returning_user'] == 0]['converted'].mean() * 100
        returning_rate = self.df_sessions[self.df_sessions['is_returning_user'] == 1]['converted'].mean() * 100
        lift = ((returning_rate / new_rate) - 1) * 100 if new_rate > 0 else 0
        insights.append(
            f"Returning users convert at {returning_rate:.2f}% vs new users at {new_rate:.2f}%, "
            f"representing a {lift:.1f}% lift - emphasizing value of retention strategies."
        )
        
        # Session engagement
        conv_duration = self.df_sessions[self.df_sessions['converted'] == 1]['session_duration_sec'].median()
        non_conv_duration = self.df_sessions[self.df_sessions['converted'] == 0]['session_duration_sec'].median()
        insights.append(
            f"Converters spend {conv_duration:.0f}s per session vs {non_conv_duration:.0f}s for non-converters, "
            f"suggesting engagement time as a conversion predictor."
        )
        
        # Product exploration
        conv_products = self.df_sessions[self.df_sessions['converted'] == 1]['unique_products_viewed'].median()
        non_conv_products = self.df_sessions[self.df_sessions['converted'] == 0]['unique_products_viewed'].median()
        insights.append(
            f"Converters view {conv_products:.0f} unique products vs {non_conv_products:.0f} for non-converters, "
            f"indicating importance of product discovery."
        )
        
        # Category insights
        if len(self.df_sessions['primary_category'].unique()) > 1:
            top_category = self.df_sessions.groupby('primary_category').agg({
                'converted': 'mean'
            }).idxmax()[0]
            top_category_rate = self.df_sessions[self.df_sessions['primary_category'] == top_category]['converted'].mean() * 100
            insights.append(
                f"Category '{top_category}' shows highest conversion at {top_category_rate:.2f}%, "
                f"suggesting category-specific optimization opportunities."
            )
        
        print("\nKEY INSIGHTS:")
        print("-" * 80)
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}\n")
        
        self.insights = insights
        return self
    
    def save_results(self, output_dir='../output'):
        """Save analysis results."""
        import os
        import numpy as np
        os.makedirs(output_dir, exist_ok=True)
        
        # Save session data
        self.df_sessions.to_csv(f'{output_dir}/session_features.csv', index=False)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert metrics to JSON-serializable format
        json_metrics = convert_numpy_types(self.conversion_metrics)
        
        # Save metrics
        with open(f'{output_dir}/conversion_metrics.json', 'w') as f:
            json.dump(json_metrics, f, indent=4)
        
        # Save insights
        with open(f'{output_dir}/insights.txt', 'w') as f:
            for i, insight in enumerate(self.insights, 1):
                f.write(f"{i}. {insight}\n\n")
        
        print(f"\n✓ Results saved to {output_dir}/")
        
        return self


# Main execution
if __name__ == "__main__":
    FILE_PATH = "../data/2019-Oct.csv"
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║                E-COMMERCE CONVERSION BEHAVIOR ANALYSIS                       ║
    ║                     Exploratory Data Analysis Pipeline                       ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create analyzer instance
    analyzer = EcommerceConversionAnalyzer(
        filepath=FILE_PATH,
        sample_size=1_000_000  # Set to None for full dataset
    )
    
    # Run full analysis pipeline
    (analyzer
     .load_data()
     .understand_data()
     .clean_and_prepare()
     .create_session_features()
     .calculate_conversion_metrics()
     .analyze_temporal_patterns()
     .analyze_behavioral_patterns()
     .analyze_category_patterns()
     .generate_insights()
     .save_results())