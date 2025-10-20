"""
A comprehensive exploratory data analysis of user behavior data and conversation
patterns in a large-scale e-commerce dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from scipy.stats import chi2_contigency, ttest_ind, f_oneway
from datetime import datetime, timedelta
import json

warnings.filterwarnings("ignore")
# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

class EcommerceConversionAnalyzer:
    """
    Main class for e-commerce conversion analysis.
    Handles data loading, cleaning, feature engineering, and analysis
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
        """Load and perform initial data inspection"""
        print("=" * 80)
        print("STEP 1: DATA LOADING")
        print("=" * 80)

        #Detect file type and load accordingly
        if self.filepath.endswith('.parquet'):
            self.df = pd.read_parquet(self.filepath)
        else:
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
        """Perform comprehensive data understanding"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA UNDERSTANDING")
        print("=" * 80)

        print("\nDataset Overview:")
        print(self.df.info())

        #Missing values
        print("\nMissing Values:")
        missing = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum().values / len(self.df)*100).round(2)
        })

        missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        print(missing)

        #Unique values
        print('\n Unique values per column')
        for col in self.df.columns:
            print(f"{col}: {self.df[col].nunique(),} unique values")
        
        #Data types
        print('\n Data types:')
        print(self.df.dtypes)

        #Statistical summary
        print("\n Statistical Summary:")
        print(self.df.describe())

        return self

    def clean_and_prepare(self):
        """Clean data and create derived features"""

        print("\n" + "=" * 80)
        print("STEP 3: DATA CLEANING & FEATURE ENGINEERING")
        print("=" * 80)

        df = self.df.copy()
        #Convert timestamps to datetime
        df['event_time'] = pd.to_datetime(df['event_time'])
        # Extract temporal features
        df['hour'] = df['event_time'].dt.hour
        df['day_of_week'] = df['event_time'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_name'] = df['event_time'].dt.day_name()
        df['date'] = df['event_time'].dt.date
        df['week'] = df['event_time'].dt.isocalendar().week
        df['month'] = df['event_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        #Time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                    bins=[0,6,12,18,24],
                                    labels=['Night','Morning','Afternoon','Evening'],
                                    include_lowest=True)
        
        #Handling missing values
        print("\n Handling missing values...")

        #Fill or drop based on column
        if 'category_code' in df.columns:
            df['category_code'] = df['category_code'].fillna('unknown')
            #Extract main category
            df['main_category'] = df['category_code'].apply(
                lambda x: x.split('.')[0] if isinstance(x, str) and x != 'unknown' else 'unknown'
            )
        
        if 'brand' in df.columns:
            df['brand'] = df['brand'].fillna('unknown')
        
        #Create user session identifier
        df['user_session'] = df['user_id'].astype(str) + '_' + df['date'].astype(str)

        #Create conversion label
        df['is_purchase'] = (df['event_type'] == 'purchase').astype(int)
        df['is_cart'] = (df['event_type'] == 'cart').astype(int)
        df['is_view'] = (df['event_type'] == 'view').astype(int)

        #Sort by user and time
        df = df.sort_values(['user_id', 'event_time'])

        print(f"✓ Created {len([c for c in df.columns if c not in self.df.columns])} new features")
        print(f"✓ New columns: {[c for c in df.columns if c not in self.df.columns]}")
        
        self.df = df
        return self
    
    def create_session_features(self):
        """Aggregate data at session level"""

        print("\n" + "=" * 80)
        print("STEP 4: SESSION-LEVEL FEATURE CREATION")
        print("=" * 80)
        
        print("\n Aggregating sessions...")

        #Session level aggregation
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
        })

        #Flatten columns
        session_agg.columns = ['_'.join(col).strip('_') for col in session_agg.columns.values]
        session_agg.columns = ['session_id', 'session_start', 'session_end', 'total_events',
                                'unique_products_viewed', 'purchases', 'carts', 'views',
                                'user_id', 'hour', 'day_of_week', 'day_name', 'is_weekend',
                                'primary_category']

        #Calculate session duration in seconds
        session_agg['session_duration_sec'] = (
            session_agg['session_end'] -  session_agg['session_start']
        ).dt.total_seconds()

        #Create conversion label
        session_agg['converted'] = (session_agg['purchases'] > 0).astype(int)

        #Cart abandonment
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
        """Calculate key conversion metrics"""
        print("\n" + "=" * 80)
        print("STEP 5: CONVERSION METRICS CALCULATION")
        print("=" * 80)

        metrics = {}

        #Overall metrics
        metrics['total_sessions'] = len(self.df_sessions)
        metrics['total_conversions'] = self.df_sessions['converted'].sum()
        metrics['overall_conversion_rate'] = (metrics['total_conversions']/metrics['total_sessions']*100)

        #Event-level metrics
        metrics['total_events'] = len(self.df)
        metrics['total_views'] = self.df['is_view'].sum()
        metrics['total_carts'] = self.df['is_cart'].sum()
        metrics['total_purchases'] = self.df['is_purchase'].sum()

        #Funnel metrics
        metrics['view_to_cart_rate'] = (metrics['total_carts'] / metrics['total_views'] * 100)
        metrics['cart_to_purchase_rate'] = (metrics['total_purchases'] / metrics['total_carts'] * 100)
        metrics['view_to_purchase_rate'] = (metrics['total_purchases'] / metrics['total_views'] * 100)

        #Cart abandonment
        metrics['cart_abandonment_count'] = self.df_sessions['cart_abandonment'].sum()
        sessions_with_carts = self.df_sessions[self.df_sessions['carts'] > 0].sum()
        metrics['cart_abandonment_rate'] = (metrics['cart_abandonment_count'] / sessions_with_carts * 100 if sessions_with_carts > 0 else 0)

        #User metrics
        metrics['total_users'] = self.df['user_id'].nunique()
        metrics['returning_users'] = self.df_sessions[self.df_sessions['is_returning_user'] == 1]['user_id'].sum()
        metrics['new_users'] = metrics['total_users'] - metrics['returning_users']
        metrics['returning_user_rate'] = (metrics['returning_users'] / metrics['total_users'] * 100)


        print("\n KEY CONVERSION METRICS:")
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
        """Analyze conversion patterns by time"""

        #Conversion by hour
        hourly = self.df_sessions.groupby('hour').agg({
            'converted': ['sum', 'mean', 'count']
        }).reset_index()
        hourly.columns = ['hour', 'conversions', 'conversion_rate', 'sessions']
        hourly['conversion_rate'] *= 100
        hourly = hourly.sort_values('conversion_rate', ascending=False)
        print("\n Top 5 hours by conversion rate:")
        print(hourly.head())

        #Conversion by day of week
        daily = self.df_sessions.groupby('dayofweek', 'day_name').agg({
            'converted': ['sum', 'mean', 'count']
        }).reset_index()
        daily.columns = ['day_of_week', 'day_name', 'conversion', 'conversion_rate', 'sessions']
        daily['conversion_rate'] *= 100
        daily = daily.sort_values('conversion_rate', ascending=False)
        print("Conversion rate by day of week:")
        print("daily")

        #Weekend vs Weekday
        weekend_analysis = self.df_sessions.groupby('is_weekend').agg({
            'converted': ['sum', 'mean', 'count']
        }).reset_index()
        weekend_analysis.columns = ['is_weekend', 'conversions', 'conversion_rate', 'sessions']
        weekend_analysis['conversion_rate'] *= 100
        weekend_analysis['period'] = weekend_analysis.map({0:'Weekday', 1:'Weekend'})
        print("Weekend vs Weekday")
        print(weekend_analysis)

        #Statistical test: weekend vs weekday
        weekday_conv = self.df_seesions[self.df_sessions['is_weekend'] == 0]['converted']
        weekend_conv = self.df_sessions[self.df_sessions['is_weekend'] == 1]['converted']
        t_stat, p_val = ttest_ind(weekend_conv, weekend_conv)

        print(f"\n T-test (Weekend vs Weekday): t={t_stat:.4f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("Statistically significant difference!")
        
        return self

    def analyze_behavioral_patterns(self):
        """Analyze user behavior patterns"""

        #Returning vs New users
        user_type_analysis = self.df_sessions.groupby('is_returning_user').agg({
            'converted': ['sum', 'mean', 'count'],
            'session_duration_rate_sec': 'mean',
            'unique_products_viewed': 'mean'
        }).reset_index()
        user_type_analysis.columns = ['is_returning', 'conversions', 'conversion_rate', 'sessions', 'avg_duration', 'avg_products_viewed']
        user_type_analysis['conversion_rate'] *= 100
        user_type_analysis['user_type'] = user_type_analysis['is_returning'].map({0:'New', 1: 'Returning'})
        print("Returning vs New users")
        print(user_type_analysis)

        #Sessions duration analysis
        converters = self.df_sessions[self.df_sessions['converted'] == 1]['session_duration_sec']
        non_converters = self.df_sessions[self.df_sessions['converted'] == 0]['session_duration_sec']

        print(f"\n Session Duration Analysis:")
        print(f"Converters: {converters.mean():.2f}s (median: {converters.median():.2f}s)")
        print(f"Non-converters: {non_converters.mean():.2f}s (median: {non_converters.median():.2f}s)")
        
        

