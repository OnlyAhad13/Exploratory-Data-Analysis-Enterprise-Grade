import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("Set2")

class ConversionVisualizer:
    """
    Comprehensive visualization suite for conversion analysis.
    Creates static (matplotlib/seaborn) and interactive (plotly) visualizations.
    """
    
    def __init__(self, df_sessions, df_events=None):
        """
        Initialize visualizer with session data.
        
        Parameters:
        -----------
        df_sessions : pd.DataFrame
            Session-level aggregated data
        df_events : pd.DataFrame, optional
            Event-level data for detailed analysis
        """
        self.df_sessions = df_sessions
        self.df_events = df_events
        self.figures = {}
        
    def create_conversion_funnel(self):
        """Create conversion funnel visualization."""
        print("Creating conversion funnel...")
        
        # Calculate funnel metrics
        total_sessions = len(self.df_sessions)
        sessions_with_views = (self.df_sessions['views'] > 0).sum()
        sessions_with_carts = (self.df_sessions['carts'] > 0).sum()
        sessions_with_purchases = (self.df_sessions['purchases'] > 0).sum()
        
        stages = ['Sessions', 'Viewed Products', 'Added to Cart', 'Purchased']
        values = [total_sessions, sessions_with_views, sessions_with_carts, sessions_with_purchases]
        
        # Calculate percentages
        percentages = [100] + [v/total_sessions*100 for v in values[1:]]
        
        # Create funnel chart with Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Funnel(
            name='Conversion Funnel',
            y=stages,
            x=values,
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(
                color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
            ),
            connector=dict(line=dict(color="royalblue", width=3))
        ))
        
        fig.update_layout(
            title='E-Commerce Conversion Funnel',
            title_font_size=20,
            height=500,
            showlegend=False
        )
        
        self.figures['funnel'] = fig
        return fig
    
    def create_temporal_heatmap(self):
        """Create hour x day of week conversion heatmap."""
        print("Creating temporal heatmap...")
        
        # Aggregate by hour and day
        heatmap_data = self.df_sessions.groupby(['day_of_week', 'hour']).agg({
            'converted': 'mean'
        }).reset_index()
        
        # Pivot for heatmap
        pivot_data = heatmap_data.pivot(index='day_of_week', columns='hour', values='converted')
        pivot_data = pivot_data * 100  # Convert to percentage
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Conversion Rate (%)'}, ax=ax,
                    yticklabels=day_names, linewidths=0.5)
        
        ax.set_title('Conversion Rate Heatmap: Hour of Day vs Day of Week', fontsize=16, pad=20)
        ax.set_xlabel('Hour of Day', fontsize=14)
        ax.set_ylabel('Day of Week', fontsize=14)
        
        plt.tight_layout()
        self.figures['temporal_heatmap'] = fig
        return fig
    
    def create_hourly_conversion_chart(self):
        """Create hourly conversion rate chart."""
        print("Creating hourly conversion chart...")
        
        # Aggregate by hour
        hourly = self.df_sessions.groupby('hour').agg({
            'converted': ['sum', 'mean', 'count']
        }).reset_index()
        hourly.columns = ['hour', 'conversions', 'conversion_rate', 'sessions']
        hourly['conversion_rate'] *= 100
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=hourly['hour'], y=hourly['sessions'], 
                   name='Sessions', marker_color='lightblue', opacity=0.6),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=hourly['hour'], y=hourly['conversion_rate'], 
                       name='Conversion Rate', mode='lines+markers',
                       marker=dict(size=10, color='red'), line=dict(width=3)),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Hour of Day", dtick=1)
        fig.update_yaxes(title_text="Number of Sessions", secondary_y=False)
        fig.update_yaxes(title_text="Conversion Rate (%)", secondary_y=True)
        
        fig.update_layout(
            title='Hourly Traffic vs Conversion Rate',
            height=500,
            hovermode='x unified',
            legend=dict(x=0.7, y=1.1, orientation='h')
        )
        
        self.figures['hourly_conversion'] = fig
        return fig
    
    def create_day_of_week_analysis(self):
        """Create day of week analysis."""
        print("Creating day of week analysis...")
        
        # Aggregate by day
        daily = self.df_sessions.groupby(['day_of_week', 'day_name']).agg({
            'converted': ['sum', 'mean', 'count']
        }).reset_index()
        daily.columns = ['day_of_week', 'day_name', 'conversions', 'conversion_rate', 'sessions']
        daily['conversion_rate'] *= 100
        daily = daily.sort_values('day_of_week')
        
        # Create subplot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Conversion rate by day
        sns.barplot(data=daily, x='day_name', y='conversion_rate', ax=axes[0], palette='viridis')
        axes[0].set_title('Conversion Rate by Day of Week', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Day of Week', fontsize=12)
        axes[0].set_ylabel('Conversion Rate (%)', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.2f%%')
        
        # Sessions by day
        sns.barplot(data=daily, x='day_name', y='sessions', ax=axes[1], palette='coolwarm')
        axes[1].set_title('Session Volume by Day of Week', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Day of Week', fontsize=12)
        axes[1].set_ylabel('Number of Sessions', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%d')
        
        plt.tight_layout()
        self.figures['day_analysis'] = fig
        return fig
    
    def create_user_type_comparison(self):
        """Create returning vs new user comparison."""
        print("Creating user type comparison...")
        
        # Aggregate by user type
        user_type = self.df_sessions.groupby('is_returning_user').agg({
            'converted': ['sum', 'mean', 'count'],
            'session_duration_sec': 'mean',
            'unique_products_viewed': 'mean',
            'carts': 'mean'
        }).reset_index()
        
        user_type.columns = ['is_returning', 'conversions', 'conversion_rate', 'sessions',
                             'avg_duration', 'avg_products', 'avg_carts']
        user_type['conversion_rate'] *= 100
        user_type['user_type'] = user_type['is_returning'].map({0: 'New Users', 1: 'Returning Users'})
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Conversion rate comparison
        sns.barplot(data=user_type, x='user_type', y='conversion_rate', ax=axes[0, 0], palette='Set2')
        axes[0, 0].set_title('Conversion Rate: New vs Returning Users', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Conversion Rate (%)', fontsize=12)
        axes[0, 0].set_xlabel('')
        for container in axes[0, 0].containers:
            axes[0, 0].bar_label(container, fmt='%.2f%%')
        
        # 2. Session duration
        sns.barplot(data=user_type, x='user_type', y='avg_duration', ax=axes[0, 1], palette='Set3')
        axes[0, 1].set_title('Average Session Duration', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Duration (seconds)', fontsize=12)
        axes[0, 1].set_xlabel('')
        for container in axes[0, 1].containers:
            axes[0, 1].bar_label(container, fmt='%.0f')
        
        # 3. Products viewed
        sns.barplot(data=user_type, x='user_type', y='avg_products', ax=axes[1, 0], palette='Pastel1')
        axes[1, 0].set_title('Average Products Viewed', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Products', fontsize=12)
        axes[1, 0].set_xlabel('')
        for container in axes[1, 0].containers:
            axes[1, 0].bar_label(container, fmt='%.1f')
        
        # 4. Cart usage
        sns.barplot(data=user_type, x='user_type', y='avg_carts', ax=axes[1, 1], palette='Set1')
        axes[1, 1].set_title('Average Cart Actions', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Cart Actions per Session', fontsize=12)
        axes[1, 1].set_xlabel('')
        for container in axes[1, 1].containers:
            axes[1, 1].bar_label(container, fmt='%.2f')
        
        plt.tight_layout()
        self.figures['user_type_comparison'] = fig
        return fig
    
    def create_session_engagement_analysis(self):
        """Analyze session engagement patterns."""
        print("Creating session engagement analysis...")
        
        # Create engagement bins
        self.df_sessions['duration_bin'] = pd.cut(
            self.df_sessions['session_duration_sec'],
            bins=[0, 30, 60, 180, 600, float('inf')],
            labels=['0-30s', '30-60s', '1-3min', '3-10min', '10min+']
        )
        
        self.df_sessions['products_bin'] = pd.cut(
            self.df_sessions['unique_products_viewed'],
            bins=[0, 1, 3, 5, 10, float('inf')],
            labels=['1', '2-3', '4-5', '6-10', '10+']
        )
        
        # Conversion by duration
        duration_conv = self.df_sessions.groupby('duration_bin').agg({
            'converted': ['mean', 'count']
        }).reset_index()
        duration_conv.columns = ['duration_bin', 'conversion_rate', 'sessions']
        duration_conv['conversion_rate'] *= 100
        
        # Conversion by products viewed
        products_conv = self.df_sessions.groupby('products_bin').agg({
            'converted': ['mean', 'count']
        }).reset_index()
        products_conv.columns = ['products_bin', 'conversion_rate', 'sessions']
        products_conv['conversion_rate'] *= 100
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Duration impact
        sns.barplot(data=duration_conv, x='duration_bin', y='conversion_rate', ax=axes[0], palette='Blues_d')
        axes[0].set_title('Conversion Rate by Session Duration', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Session Duration', fontsize=12)
        axes[0].set_ylabel('Conversion Rate (%)', fontsize=12)
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.2f%%')
        
        # Products viewed impact
        sns.barplot(data=products_conv, x='products_bin', y='conversion_rate', ax=axes[1], palette='Greens_d')
        axes[1].set_title('Conversion Rate by Products Viewed', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Products Viewed', fontsize=12)
        axes[1].set_ylabel('Conversion Rate (%)', fontsize=12)
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%.2f%%')
        
        plt.tight_layout()
        self.figures['engagement_analysis'] = fig
        return fig
    
    def create_category_analysis(self):
        """Create category performance analysis."""
        print("Creating category analysis...")
        
        # Aggregate by category
        category = self.df_sessions.groupby('primary_category').agg({
            'converted': ['sum', 'mean', 'count']
        }).reset_index()
        category.columns = ['category', 'conversions', 'conversion_rate', 'sessions']
        category['conversion_rate'] *= 100
        
        # Filter categories with sufficient volume
        category = category[category['sessions'] >= 100]
        category = category.sort_values('conversion_rate', ascending=False).head(15)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(14, 10))
        
        sns.barplot(data=category, y='category', x='conversion_rate', ax=ax, palette='rocket')
        ax.set_title('Top 15 Categories by Conversion Rate', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Conversion Rate (%)', fontsize=12)
        ax.set_ylabel('Product Category', fontsize=12)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f%%', padding=3)
        
        plt.tight_layout()
        self.figures['category_analysis'] = fig
        return fig
    
    def create_cart_abandonment_analysis(self):
        """Analyze cart abandonment patterns."""
        print("Creating cart abandonment analysis...")
        
        # Filter sessions with cart activity
        cart_sessions = self.df_sessions[self.df_sessions['carts'] > 0].copy()
        
        if len(cart_sessions) == 0:
            print("No cart sessions found.")
            return None
        
        # Abandonment by hour
        abandonment_hourly = cart_sessions.groupby('hour').agg({
            'cart_abandonment': 'mean',
            'session_id': 'count'
        }).reset_index()
        abandonment_hourly.columns = ['hour', 'abandonment_rate', 'sessions']
        abandonment_hourly['abandonment_rate'] *= 100
        
        # Abandonment by day
        abandonment_daily = cart_sessions.groupby('day_name').agg({
            'cart_abandonment': 'mean',
            'session_id': 'count'
        }).reset_index()
        abandonment_daily.columns = ['day_name', 'abandonment_rate', 'sessions']
        abandonment_daily['abandonment_rate'] *= 100
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Hourly abandonment
        axes[0].plot(abandonment_hourly['hour'], abandonment_hourly['abandonment_rate'], 
                     marker='o', linewidth=2, markersize=8, color='crimson')
        axes[0].fill_between(abandonment_hourly['hour'], abandonment_hourly['abandonment_rate'], 
                            alpha=0.3, color='crimson')
        axes[0].set_title('Cart Abandonment Rate by Hour', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Hour of Day', fontsize=12)
        axes[0].set_ylabel('Abandonment Rate (%)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Daily abandonment
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        abandonment_daily['day_name'] = pd.Categorical(abandonment_daily['day_name'], 
                                                        categories=day_order, ordered=True)
        abandonment_daily = abandonment_daily.sort_values('day_name')
        
        sns.barplot(data=abandonment_daily, x='day_name', y='abandonment_rate', 
                   ax=axes[1], palette='Reds_d')
        axes[1].set_title('Cart Abandonment Rate by Day', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Day of Week', fontsize=12)
        axes[1].set_ylabel('Abandonment Rate (%)', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%.1f%%')
        
        plt.tight_layout()
        self.figures['cart_abandonment'] = fig
        return fig
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap of key metrics."""
        print("Creating correlation heatmap...")
        
        # Select numeric columns for correlation
        corr_cols = ['session_duration_sec', 'unique_products_viewed', 'total_events',
                     'views', 'carts', 'purchases', 'converted', 'is_returning_user',
                     'is_weekend', 'hour']
        
        # Filter existing columns
        corr_cols = [col for col in corr_cols if col in self.df_sessions.columns]
        
        # Calculate correlation matrix
        corr_matrix = self.df_sessions[corr_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Correlation Matrix: Key Conversion Metrics', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        self.figures['correlation_heatmap'] = fig
        return fig
    
    def create_conversion_distribution(self):
        """Create distribution plots for converters vs non-converters."""
        print("Creating conversion distribution plots...")
        
        converters = self.df_sessions[self.df_sessions['converted'] == 1]
        non_converters = self.df_sessions[self.df_sessions['converted'] == 0]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Session duration distribution
        axes[0, 0].hist([non_converters['session_duration_sec'].clip(upper=1000), 
                        converters['session_duration_sec'].clip(upper=1000)],
                       bins=50, label=['Non-converters', 'Converters'], alpha=0.7)
        axes[0, 0].set_title('Session Duration Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Duration (seconds, capped at 1000)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].legend()
        
        # Products viewed distribution
        axes[0, 1].hist([non_converters['unique_products_viewed'].clip(upper=20), 
                        converters['unique_products_viewed'].clip(upper=20)],
                       bins=20, label=['Non-converters', 'Converters'], alpha=0.7)
        axes[0, 1].set_title('Products Viewed Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Number of Products (capped at 20)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend()
        
        # Total events distribution
        axes[1, 0].hist([non_converters['total_events'].clip(upper=50), 
                        converters['total_events'].clip(upper=50)],
                       bins=25, label=['Non-converters', 'Converters'], alpha=0.7)
        axes[1, 0].set_title('Total Events Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Events (capped at 50)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].legend()
        
        # Box plot: session duration
        data_to_plot = [non_converters['session_duration_sec'].clip(upper=1000),
                        converters['session_duration_sec'].clip(upper=1000)]
        axes[1, 1].boxplot(data_to_plot, labels=['Non-converters', 'Converters'])
        axes[1, 1].set_title('Session Duration: Box Plot Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Duration (seconds, capped at 1000)', fontsize=12)
        
        plt.tight_layout()
        self.figures['distribution_analysis'] = fig
        return fig
    
    def create_dashboard_summary(self):
        """Create comprehensive dashboard summary."""
        print("Creating dashboard summary...")
        
        # Calculate key metrics
        total_sessions = len(self.df_sessions)
        total_conversions = self.df_sessions['converted'].sum()
        conversion_rate = (total_conversions / total_sessions * 100)
        avg_duration = self.df_sessions['session_duration_sec'].mean()
        avg_products = self.df_sessions['unique_products_viewed'].mean()
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('E-Commerce Conversion Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Metric cards
        metrics = [
            ('Total Sessions', f'{total_sessions:,}', '#3498db'),
            ('Conversions', f'{total_conversions:,}', '#2ecc71'),
            ('Conversion Rate', f'{conversion_rate:.2f}%', '#e74c3c'),
            ('Avg Duration', f'{avg_duration:.0f}s', '#f39c12'),
            ('Avg Products', f'{avg_products:.1f}', '#9b59b6')
        ]
        
        for i, (label, value, color) in enumerate(metrics[:3]):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.6, value, ha='center', va='center', 
                   fontsize=32, fontweight='bold', color=color)
            ax.text(0.5, 0.2, label, ha='center', va='center', 
                   fontsize=14, color='gray')
            ax.axis('off')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                      fill=False, edgecolor=color, linewidth=3))
        
        # Hourly trend
        ax1 = fig.add_subplot(gs[1, :2])
        hourly = self.df_sessions.groupby('hour')['converted'].mean() * 100
        ax1.plot(hourly.index, hourly.values, marker='o', linewidth=2, markersize=6)
        ax1.fill_between(hourly.index, hourly.values, alpha=0.3)
        ax1.set_title('Conversion Rate by Hour', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Conversion Rate (%)')
        ax1.grid(True, alpha=0.3)
        
        # Day of week
        ax2 = fig.add_subplot(gs[1, 2])
        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_conv = self.df_sessions.groupby('day_of_week')['converted'].mean() * 100
        ax2.barh(range(len(daily_conv)), daily_conv.values, color='skyblue')
        ax2.set_yticks(range(7))
        ax2.set_yticklabels(day_order)
        ax2.set_title('Conversion by Day', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Conv. Rate (%)')
        
        # User type comparison
        ax3 = fig.add_subplot(gs[2, 0])
        user_conv = self.df_sessions.groupby('is_returning_user')['converted'].mean() * 100
        colors = ['#e74c3c', '#2ecc71']
        ax3.bar(['New', 'Returning'], user_conv.values, color=colors, alpha=0.7)
        ax3.set_title('New vs Returning', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Conversion Rate (%)')
        
        # Top categories
        ax4 = fig.add_subplot(gs[2, 1:])
        top_cats = self.df_sessions.groupby('primary_category').agg({
            'converted': 'mean', 'session_id': 'count'
        })
        top_cats = top_cats[top_cats['session_id'] >= 100].sort_values('converted', ascending=False).head(8)
        top_cats['converted'] *= 100
        ax4.barh(range(len(top_cats)), top_cats['converted'].values, color='coral')
        ax4.set_yticks(range(len(top_cats)))
        ax4.set_yticklabels([cat[:20] for cat in top_cats.index])
        ax4.set_title('Top Categories by Conversion', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Conversion Rate (%)')
        
        self.figures['dashboard'] = fig
        return fig
    
    def save_all_figures(self, output_dir='../output/visualizations'):
        """Save all generated figures."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving visualizations to {output_dir}/")
        
        for name, fig in self.figures.items():
            if isinstance(fig, go.Figure):
                # Save Plotly figures as HTML
                fig.write_html(f'{output_dir}/{name}.html')
                print(f" Saved {name}.html")
            else:
                # Save matplotlib figures as PNG
                fig.savefig(f'{output_dir}/{name}.png', dpi=300, bbox_inches='tight')
                print(f" Saved {name}.png")
                plt.close(fig)
        
        print(f"\nAll visualizations saved!")
        
    def generate_all_visualizations(self):
        """Generate all visualizations in sequence."""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE VISUALIZATION SUITE")
        print("=" * 80 + "\n")
        
        self.create_conversion_funnel()
        self.create_temporal_heatmap()
        self.create_hourly_conversion_chart()
        self.create_day_of_week_analysis()
        self.create_user_type_comparison()
        self.create_session_engagement_analysis()
        self.create_category_analysis()
        self.create_cart_abandonment_analysis()
        self.create_correlation_heatmap()
        self.create_conversion_distribution()
        self.create_dashboard_summary()
        
        print("\nAll visualizations created!")
        
        return self


# Main execution
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║              E-COMMERCE CONVERSION VISUALIZATION SUITE                       ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Load session data
    print("Loading session data...")
    df_sessions = pd.read_csv('../output/session_features.csv')
    
    # Create visualizer
    visualizer = ConversionVisualizer(df_sessions)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    # Save figures
    visualizer.save_all_figures()
    
    print("COMPLETED")