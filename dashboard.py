"""
E-Commerce Conversion Analysis Dashboard
=======================================
Comprehensive Streamlit dashboard for e-commerce conversion analysis.
Integrates main analysis, visualization, and statistical analysis modules.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime
import warnings

# Add src directory to path
sys.path.append('src')

# Import our analysis modules
from main_analysis import EcommerceConversionAnalyzer
from visualization import ConversionVisualizer
from statistical_analysis import StatisticalAnalyzer

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="E-Commerce Conversion Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .error-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_session_data():
    """Load session data with caching."""
    try:
        df_sessions = pd.read_csv('output/session_features.csv')
        return df_sessions
    except FileNotFoundError:
        st.error("Session data not found. Please run the main analysis first.")
        return None

@st.cache_data
def load_conversion_metrics():
    """Load conversion metrics with caching."""
    try:
        with open('output/conversion_metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_statistical_results():
    """Load statistical test results with caching."""
    try:
        with open('output/statistical_tests.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 E-Commerce Conversion Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Data loading status
    st.sidebar.subheader("📁 Data Status")
    df_sessions = load_session_data()
    conversion_metrics = load_conversion_metrics()
    statistical_results = load_statistical_results()
    
    if df_sessions is not None:
        st.sidebar.success(f"✅ Session data loaded ({len(df_sessions):,} sessions)")
    else:
        st.sidebar.error("❌ Session data not found")
        return
    
    if conversion_metrics is not None:
        st.sidebar.success("✅ Conversion metrics loaded")
    else:
        st.sidebar.warning("⚠️ Conversion metrics not found")
    
    if statistical_results is not None:
        st.sidebar.success("✅ Statistical results loaded")
    else:
        st.sidebar.warning("⚠️ Statistical results not found")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Main Analysis", "📊 Visualizations", "🧮 Statistical Analysis"])
    
    with tab1:
        main_analysis_tab(df_sessions, conversion_metrics)
    
    with tab2:
        visualization_tab(df_sessions)
    
    with tab3:
        statistical_analysis_tab(df_sessions, statistical_results)

def main_analysis_tab(df_sessions, conversion_metrics):
    """Main analysis tab content."""
    st.header("📈 Main Analysis Dashboard")
    
    # Key metrics overview
    st.subheader("🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sessions = len(df_sessions)
        st.metric("Total Sessions", f"{total_sessions:,}")
    
    with col2:
        conversion_rate = df_sessions['converted'].mean() * 100
        st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    
    with col3:
        avg_duration = df_sessions['session_duration_sec'].mean() / 60  # Convert to minutes
        st.metric("Avg Session Duration", f"{avg_duration:.1f} min")
    
    with col4:
        total_events = df_sessions['total_events'].sum()
        st.metric("Total Events", f"{total_events:,}")
    
    # Data overview
    st.subheader("📋 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Session Data Summary:**")
        st.dataframe(df_sessions.describe(), use_container_width=True)
    
    with col2:
        st.write("**Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': df_sessions.columns,
            'Type': df_sessions.dtypes.astype(str),
            'Non-Null Count': df_sessions.count(),
            'Null Count': df_sessions.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Conversion funnel
    st.subheader("🔄 Conversion Funnel")
    
    funnel_data = {
        'Stage': ['Sessions', 'Viewed Products', 'Added to Cart', 'Purchased'],
        'Count': [
            len(df_sessions),
            (df_sessions['views'] > 0).sum(),
            (df_sessions['carts'] > 0).sum(),
            (df_sessions['purchases'] > 0).sum()
        ]
    }
    
    funnel_df = pd.DataFrame(funnel_data)
    funnel_df['Conversion Rate'] = (funnel_df['Count'] / funnel_df['Count'].iloc[0] * 100).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(funnel_df, use_container_width=True)
    
    with col2:
        fig = px.funnel(funnel_df, x='Count', y='Stage', 
                       title="Conversion Funnel")
        st.plotly_chart(fig, use_container_width=True)
    
    # Temporal patterns
    st.subheader("⏰ Temporal Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly conversion rates
        hourly_conv = df_sessions.groupby('hour')['converted'].mean() * 100
        fig_hourly = px.line(x=hourly_conv.index, y=hourly_conv.values,
                           title="Conversion Rate by Hour",
                           labels={'x': 'Hour of Day', 'y': 'Conversion Rate (%)'})
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Day analysis
        if 'day_name' in df_sessions.columns:
            daily_conv = df_sessions.groupby('day_name')['converted'].mean() * 100
            fig_daily = px.bar(x=daily_conv.index, y=daily_conv.values,
                             title="Conversion Rate by Day",
                             labels={'x': 'Day of Week', 'y': 'Conversion Rate (%)'})
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.info("Day of week data not available")
    
    # Category analysis
    st.subheader("🏷️ Category Analysis")
    
    if 'primary_category' in df_sessions.columns:
        category_conv = df_sessions.groupby('primary_category').agg({
            'converted': ['mean', 'count']
        }).round(4)
        category_conv.columns = ['Conversion Rate', 'Session Count']
        category_conv['Conversion Rate'] *= 100
        category_conv = category_conv.sort_values('Conversion Rate', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Categories by Conversion Rate:**")
            st.dataframe(category_conv.head(10), use_container_width=True)
        
        with col2:
            fig_cat = px.bar(category_conv.head(10), 
                           x=category_conv.head(10).index, 
                           y='Conversion Rate',
                           title="Top 10 Categories by Conversion Rate")
            st.plotly_chart(fig_cat, use_container_width=True)

def visualization_tab(df_sessions):
    """Visualization tab content."""
    st.header("📊 Interactive Visualizations")
    
    # Initialize visualizer
    visualizer = ConversionVisualizer(df_sessions)
    
    # Visualization options
    st.subheader("🎨 Available Visualizations")
    
    viz_options = [
        "Conversion Funnel",
        "Temporal Heatmap", 
        "Hourly Conversion Chart",
        "Day of Week Analysis",
        "User Type Comparison",
        "Session Engagement Analysis",
        "Category Analysis",
        "Cart Abandonment Analysis",
        "Correlation Heatmap",
        "Conversion Distribution",
        "Dashboard Summary"
    ]
    
    selected_viz = st.selectbox("Select Visualization:", viz_options)
    
    # Generate selected visualization
    if st.button("Generate Visualization", type="primary"):
        with st.spinner(f"Generating {selected_viz}..."):
            try:
                if selected_viz == "Conversion Funnel":
                    fig = visualizer.create_conversion_funnel()
                    st.plotly_chart(fig, use_container_width=True)
                
                elif selected_viz == "Temporal Heatmap":
                    fig = visualizer.create_temporal_heatmap()
                    st.pyplot(fig)
                
                elif selected_viz == "Hourly Conversion Chart":
                    fig = visualizer.create_hourly_conversion_chart()
                    st.plotly_chart(fig, use_container_width=True)
                
                elif selected_viz == "Day of Week Analysis":
                    fig = visualizer.create_day_of_week_analysis()
                    st.pyplot(fig)
                
                elif selected_viz == "User Type Comparison":
                    fig = visualizer.create_user_type_comparison()
                    st.pyplot(fig)
                
                elif selected_viz == "Session Engagement Analysis":
                    fig = visualizer.create_session_engagement_analysis()
                    st.pyplot(fig)
                
                elif selected_viz == "Category Analysis":
                    fig = visualizer.create_category_analysis()
                    st.pyplot(fig)
                
                elif selected_viz == "Cart Abandonment Analysis":
                    fig = visualizer.create_cart_abandonment_analysis()
                    st.pyplot(fig)
                
                elif selected_viz == "Correlation Heatmap":
                    fig = visualizer.create_correlation_heatmap()
                    st.pyplot(fig)
                
                elif selected_viz == "Conversion Distribution":
                    fig = visualizer.create_conversion_distribution()
                    st.pyplot(fig)
                
                elif selected_viz == "Dashboard Summary":
                    fig = visualizer.create_dashboard_summary()
                    st.pyplot(fig)
                
                st.success(f"✅ {selected_viz} generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating {selected_viz}: {str(e)}")
    
    # Pre-generated visualizations
    st.subheader("🖼️ Pre-generated Visualizations")
    
    viz_dir = "output/visualizations"
    if os.path.exists(viz_dir):
        viz_files = [f for f in os.listdir(viz_dir) if f.endswith(('.png', '.html'))]
        
        if viz_files:
            selected_file = st.selectbox("Select pre-generated visualization:", viz_files)
            
            if selected_file.endswith('.html'):
                with open(os.path.join(viz_dir, selected_file), 'r') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600)
            else:
                st.image(os.path.join(viz_dir, selected_file), use_column_width=True)
        else:
            st.info("No pre-generated visualizations found. Run the visualization module first.")
    else:
        st.info("Visualizations directory not found.")

def statistical_analysis_tab(df_sessions, statistical_results):
    """Statistical analysis tab content."""
    st.header("🧮 Statistical Analysis Dashboard")
    
    if statistical_results is None:
        st.error("Statistical results not found. Please run the statistical analysis first.")
        return
    
    # Key statistical findings
    st.subheader("📊 Key Statistical Findings")
    
    # Temporal effects
    if 'temporal' in statistical_results:
        st.write("**⏰ Temporal Effects:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hour_effect = statistical_results['temporal']['hour_anova']
            if hour_effect['significant']:
                st.success(f"✅ Hour of day significantly affects conversion (F={hour_effect['f_stat']:.2f}, p={hour_effect['p_value']:.2e})")
            else:
                st.info("ℹ️ Hour of day does not significantly affect conversion")
        
        with col2:
            day_effect = statistical_results['temporal']['day_anova']
            if day_effect.get('insufficient_data'):
                st.warning("⚠️ Insufficient data for day-of-week analysis")
            elif day_effect['significant']:
                st.success(f"✅ Day of week significantly affects conversion")
            else:
                st.info("ℹ️ Day of week does not significantly affect conversion")
        
        with col3:
            weekend_effect = statistical_results['temporal']['weekend_ttest']
            if weekend_effect['significant']:
                st.success("✅ Weekend vs weekday significantly affects conversion")
            else:
                st.info("ℹ️ Weekend vs weekday does not significantly affect conversion")
    
    # User type effects
    if 'user_type' in statistical_results:
        st.write("**👥 User Type Effects:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            conv_effect = statistical_results['user_type']['conversion']
            if conv_effect.get('insufficient_data'):
                st.warning("⚠️ Insufficient data for user type comparison")
            elif conv_effect['significant']:
                st.success("✅ User type significantly affects conversion")
            else:
                st.info("ℹ️ User type does not significantly affect conversion")
        
        with col2:
            duration_effect = statistical_results['user_type']['duration']
            if duration_effect.get('insufficient_data'):
                st.warning("⚠️ Insufficient data for duration comparison")
            elif duration_effect['significant']:
                st.success("✅ User type significantly affects session duration")
            else:
                st.info("ℹ️ User type does not significantly affect session duration")
        
        with col3:
            products_effect = statistical_results['user_type']['products']
            if products_effect.get('insufficient_data'):
                st.warning("⚠️ Insufficient data for products comparison")
            elif products_effect['significant']:
                st.success("✅ User type significantly affects products viewed")
            else:
                st.info("ℹ️ User type does not significantly affect products viewed")
    
    # Category effects
    if 'category' in statistical_results:
        st.write("**🏷️ Category Effects:**")
        
        cat_effect = statistical_results['category']
        if cat_effect['significant']:
            st.success(f"✅ Category significantly affects conversion (χ²={cat_effect['chi2']:.2f}, p={cat_effect['p_value']:.2e})")
            st.write(f"**Effect Size (Cramér's V):** {cat_effect['cramers_v']:.4f}")
        else:
            st.info("ℹ️ Category does not significantly affect conversion")
    
    # Engagement effects
    if 'engagement' in statistical_results:
        st.write("**🎯 Engagement Effects:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            duration_eng = statistical_results['engagement']['duration']
            if duration_eng['significant']:
                st.success("✅ Session duration significantly affects conversion")
            else:
                st.info("ℹ️ Session duration does not significantly affect conversion")
        
        with col2:
            products_eng = statistical_results['engagement']['products']
            if products_eng['significant']:
                st.success("✅ Products viewed significantly affects conversion")
            else:
                st.info("ℹ️ Products viewed does not significantly affect conversion")
        
        with col3:
            events_eng = statistical_results['engagement']['events']
            if events_eng['significant']:
                st.success("✅ Total events significantly affects conversion")
            else:
                st.info("ℹ️ Total events does not significantly affect conversion")
    
    # Correlation analysis
    if 'correlations' in statistical_results:
        st.write("**🔗 Correlation Analysis:**")
        
        correlations = statistical_results['correlations']
        
        # Create correlation summary table
        corr_data = []
        for var, stats in correlations.items():
            corr_data.append({
                'Variable': var,
                'Pearson r': f"{stats['pearson_r']:.4f}",
                'Spearman ρ': f"{stats['spearman_r']:.4f}",
                'Significance': "✅ Significant" if stats['significant'] else "❌ Not Significant"
            })
        
        corr_df = pd.DataFrame(corr_data)
        st.dataframe(corr_df, use_container_width=True)
    
    # Detailed results
    st.subheader("📋 Detailed Statistical Results")
    
    if st.checkbox("Show detailed JSON results"):
        st.json(statistical_results)
    
    # Run new analysis
    st.subheader("🔄 Run New Statistical Analysis")
    
    if st.button("Run Statistical Analysis", type="primary"):
        with st.spinner("Running statistical analysis..."):
            try:
                analyzer = StatisticalAnalyzer(df_sessions)
                results, summary = analyzer.run_full_analysis()
                st.success("✅ Statistical analysis completed successfully!")
                st.write("**Summary:**", summary)
            except Exception as e:
                st.error(f"❌ Error running statistical analysis: {str(e)}")

if __name__ == "__main__":
    main()
