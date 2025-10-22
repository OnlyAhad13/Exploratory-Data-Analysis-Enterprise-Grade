import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, mannwhitneyu
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for conversion data.
    Performs hypothesis testing, effect size calculations, and significance analysis.
    """
    
    def __init__(self, df_sessions):
        """
        Initialize with session data.
        
        Parameters:
        -----------
        df_sessions : pd.DataFrame
            Session-level aggregated data
        """
        self.df_sessions = df_sessions
        self.test_results = {}
        
    def calculate_cohens_d(self, group1, group2):
        """
        Calculate Cohen's d effect size.
        
        Parameters:
        -----------
        group1, group2 : array-like
            Two groups to compare
            
        Returns:
        --------
        float : Cohen's d value
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        if pooled_std == 0:
            return 0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def calculate_cramers_v(self, contingency_table):
        """
        Calculate Cramér's V effect size for categorical association.
        
        Parameters:
        -----------
        contingency_table : pd.DataFrame
            Contingency table
            
        Returns:
        --------
        float : Cramér's V value
        """
        chi2 = chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        
        if n == 0 or min_dim == 0:
            return 0
        
        return np.sqrt(chi2 / (n * min_dim))
    
    def interpret_cramers_v(self, v):
        """Interpret Cramér's V effect size."""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    
    def test_temporal_effects(self):
        """Test temporal patterns on conversion."""
        print("\n" + "=" * 80)
        print("TEMPORAL EFFECTS ANALYSIS")
        print("=" * 80)
        
        results = {}
        
        # 1. Hour of day effect
        print("\n1. HOUR OF DAY EFFECT")
        print("-" * 80)
        
        # Group sessions by hour
        hourly_groups = [group['converted'].values 
                        for _, group in self.df_sessions.groupby('hour')]
        
        # One-way ANOVA
        f_stat, p_value = f_oneway(*hourly_groups)
        
        print(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.6f}")
        
        if p_value < 0.05:
            print("✓ Hour of day has SIGNIFICANT effect on conversion")
            
            # Find best and worst hours
            hourly_conv = self.df_sessions.groupby('hour')['converted'].mean()
            best_hour = hourly_conv.idxmax()
            worst_hour = hourly_conv.idxmin()
            
            print(f"\nBest hour: {best_hour}:00 ({hourly_conv[best_hour]*100:.2f}% conversion)")
            print(f"Worst hour: {worst_hour}:00 ({hourly_conv[worst_hour]*100:.2f}% conversion)")
            print(f"Relative lift: {(hourly_conv[best_hour]/hourly_conv[worst_hour]-1)*100:.1f}%")
        else:
            print("✗ Hour of day does NOT have significant effect")
        
        results['hour_anova'] = {'f_stat': f_stat, 'p_value': p_value, 
                                 'significant': p_value < 0.05}
        
        # 2. Day of week effect
        print("\n2. DAY OF WEEK EFFECT")
        print("-" * 80)
        
        # Check if we have data for multiple days
        unique_days = self.df_sessions['day_of_week'].nunique()
        
        if unique_days < 2:
            print(f"Only {unique_days} day(s) of data available - cannot test day-of-week effects")
            print("Day-of-week analysis requires data from multiple days")
            results['day_anova'] = {'f_stat': None, 'p_value': None,
                                   'significant': False, 'insufficient_data': True}
        else:
            daily_groups = [group['converted'].values 
                           for _, group in self.df_sessions.groupby('day_of_week')]
            
            f_stat, p_value = f_oneway(*daily_groups)
            
            print(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.6f}")
            
            if p_value < 0.05:
                print("✓ Day of week has SIGNIFICANT effect on conversion")
                
                daily_conv = self.df_sessions.groupby('day_name')['converted'].mean()
                best_day = daily_conv.idxmax()
                worst_day = daily_conv.idxmin()
                
                print(f"\nBest day: {best_day} ({daily_conv[best_day]*100:.2f}% conversion)")
                print(f"Worst day: {worst_day} ({daily_conv[worst_day]*100:.2f}% conversion)")
            else:
                print("✗ Day of week does NOT have significant effect")
            
            results['day_anova'] = {'f_stat': f_stat, 'p_value': p_value,
                                   'significant': p_value < 0.05}
        
        # 3. Weekend vs Weekday
        print("\n3. WEEKEND VS WEEKDAY EFFECT")
        print("-" * 80)
        
        weekday = self.df_sessions[self.df_sessions['is_weekend'] == 0]['converted']
        weekend = self.df_sessions[self.df_sessions['is_weekend'] == 1]['converted']
        
        # T-test
        t_stat, p_value = ttest_ind(weekday, weekend)
        
        # Effect size
        cohens_d = self.calculate_cohens_d(weekend, weekday)
        
        print(f"Independent t-test: t={t_stat:.4f}, p={p_value:.6f}")
        print(f"Cohen's d: {cohens_d:.4f} ({self.interpret_cohens_d(cohens_d)} effect)")
        print(f"\nWeekday conversion: {weekday.mean()*100:.2f}%")
        print(f"Weekend conversion: {weekend.mean()*100:.2f}%")
        print(f"Difference: {(weekend.mean() - weekday.mean())*100:.2f} percentage points")
        
        if p_value < 0.05:
            print("✓ Statistically significant difference")
        else:
            print("✗ No significant difference")
        
        results['weekend_ttest'] = {
            't_stat': t_stat, 'p_value': p_value,
            'cohens_d': cohens_d, 'significant': p_value < 0.05
        }
        
        self.test_results['temporal'] = results
        return results
    
    def test_user_type_effects(self):
        """Test user type (new vs returning) effects."""
        print("\n" + "=" * 80)
        print("USER TYPE EFFECTS ANALYSIS")
        print("=" * 80)
        
        results = {}
        
        new_users = self.df_sessions[self.df_sessions['is_returning_user'] == 0]
        returning_users = self.df_sessions[self.df_sessions['is_returning_user'] == 1]
        
        # Check if we have both user types
        if len(returning_users) == 0:
            print("\n1. CONVERSION RATE COMPARISON")
            print("-" * 80)
            print(" No returning users found - cannot compare user types")
            print(f"   All {len(new_users)} users are new users")
            results['conversion'] = {
                't_stat': None, 'p_value': None,
                'cohens_d': None, 'significant': False, 'insufficient_data': True
            }
        else:
            # 1. Conversion rate difference
            print("\n1. CONVERSION RATE COMPARISON")
            print("-" * 80)
            
            new_conv = new_users['converted']
            returning_conv = returning_users['converted']
            
            t_stat, p_value = ttest_ind(returning_conv, new_conv)
            cohens_d = self.calculate_cohens_d(returning_conv, new_conv)
            
            print(f"Independent t-test: t={t_stat:.4f}, p={p_value:.6f}")
            print(f"Cohen's d: {cohens_d:.4f} ({self.interpret_cohens_d(cohens_d)} effect)")
            print(f"\nNew users conversion: {new_conv.mean()*100:.2f}%")
            print(f"Returning users conversion: {returning_conv.mean()*100:.2f}%")
            print(f"Lift: {(returning_conv.mean()/new_conv.mean()-1)*100:.1f}%")
            
            if p_value < 0.05:
                print("✓ Statistically significant difference")
            
            results['conversion'] = {
                't_stat': t_stat, 'p_value': p_value,
                'cohens_d': cohens_d, 'significant': p_value < 0.05
            }
        
        # 2. Session duration difference
        print("\n2. SESSION DURATION COMPARISON")
        print("-" * 80)
        
        if len(returning_users) == 0:
            print("No returning users found - cannot compare durations")
            print(f"   New users median duration: {new_users['session_duration_sec'].median():.0f}s")
            results['duration'] = {
                'u_stat': None, 'p_value': None,
                'significant': False, 'insufficient_data': True
            }
        else:
            new_duration = new_users['session_duration_sec']
            returning_duration = returning_users['session_duration_sec']
            
            # Use Mann-Whitney U test (non-parametric) for duration
            u_stat, p_value = mannwhitneyu(returning_duration, new_duration, alternative='two-sided')
            
            print(f"Mann-Whitney U test: U={u_stat:.4f}, p={p_value:.6f}")
            print(f"\nNew users median duration: {new_duration.median():.0f}s")
            print(f"Returning users median duration: {returning_duration.median():.0f}s")
            
            if p_value < 0.05:
                print("✓ Statistically significant difference")
            
            results['duration'] = {
                'u_stat': u_stat, 'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # 3. Products viewed difference
        print("\n3. PRODUCTS VIEWED COMPARISON")
        print("-" * 80)
        
        if len(returning_users) == 0:
            print("No returning users found - cannot compare products viewed")
            print(f"   New users median products: {new_users['unique_products_viewed'].median():.0f}")
            results['products'] = {
                'u_stat': None, 'p_value': None,
                'significant': False, 'insufficient_data': True
            }
        else:
            new_products = new_users['unique_products_viewed']
            returning_products = returning_users['unique_products_viewed']
            
            u_stat, p_value = mannwhitneyu(returning_products, new_products, alternative='two-sided')
            
            print(f"Mann-Whitney U test: U={u_stat:.4f}, p={p_value:.6f}")
            print(f"\nNew users median products: {new_products.median():.0f}")
            print(f"Returning users median products: {returning_products.median():.0f}")
            
            if p_value < 0.05:
                print("✓ Statistically significant difference")
            
            results['products'] = {
                'u_stat': u_stat, 'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        self.test_results['user_type'] = results
        return results
    
    def test_category_effects(self):
        """Test product category effects on conversion."""
        print("\n" + "=" * 80)
        print("CATEGORY EFFECTS ANALYSIS")
        print("=" * 80)
        
        # Create contingency table
        contingency = pd.crosstab(
            self.df_sessions['primary_category'],
            self.df_sessions['converted']
        )
        
        # Filter categories with at least 50 sessions
        category_counts = self.df_sessions['primary_category'].value_counts()
        valid_categories = category_counts[category_counts >= 50].index
        contingency = contingency.loc[valid_categories]
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        cramers_v = self.calculate_cramers_v(contingency)
        
        print("\nChi-Square Test of Independence")
        print("-" * 80)
        print(f"χ² statistic: {chi2:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Degrees of freedom: {dof}")
        print(f"Cramér's V: {cramers_v:.4f} ({self.interpret_cramers_v(cramers_v)} effect)")
        
        if p_value < 0.05:
            print("\n✓ Category has SIGNIFICANT effect on conversion")
            
            # Show top and bottom categories
            category_conv = self.df_sessions.groupby('primary_category').agg({
                'converted': ['mean', 'count']
            })
            category_conv.columns = ['conversion_rate', 'sessions']
            category_conv = category_conv[category_conv['sessions'] >= 50]
            category_conv = category_conv.sort_values('conversion_rate', ascending=False)
            
            print(f"\nTop 5 categories:")
            for cat, row in category_conv.head().iterrows():
                print(f"  {cat}: {row['conversion_rate']*100:.2f}% ({int(row['sessions'])} sessions)")
            
            print(f"\nBottom 5 categories:")
            for cat, row in category_conv.tail().iterrows():
                print(f"  {cat}: {row['conversion_rate']*100:.2f}% ({int(row['sessions'])} sessions)")
        else:
            print("\n✗ Category does NOT have significant effect")
        
        results = {
            'chi2': chi2, 'p_value': p_value, 'dof': dof,
            'cramers_v': cramers_v, 'significant': p_value < 0.05
        }
        
        self.test_results['category'] = results
        return results
    
    def test_engagement_effects(self):
        """Test engagement metrics on conversion."""
        print("\n" + "=" * 80)
        print("ENGAGEMENT EFFECTS ANALYSIS")
        print("=" * 80)
        
        results = {}
        
        converters = self.df_sessions[self.df_sessions['converted'] == 1]
        non_converters = self.df_sessions[self.df_sessions['converted'] == 0]
        
        # 1. Session duration effect
        print("\n1. SESSION DURATION EFFECT")
        print("-" * 80)
        
        conv_duration = converters['session_duration_sec']
        non_conv_duration = non_converters['session_duration_sec']
        
        # Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(conv_duration, non_conv_duration, alternative='two-sided')
        
        # Calculate medians and means
        print(f"Mann-Whitney U test: U={u_stat:.4f}, p={p_value:.6f}")
        print(f"\nConverters:")
        print(f"  Mean: {conv_duration.mean():.2f}s")
        print(f"  Median: {conv_duration.median():.2f}s")
        print(f"\nNon-converters:")
        print(f"  Mean: {non_conv_duration.mean():.2f}s")
        print(f"  Median: {non_conv_duration.median():.2f}s")
        
        if p_value < 0.05:
            print("\n✓ Duration has SIGNIFICANT effect on conversion")
        
        results['duration'] = {
            'u_stat': u_stat, 'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # 2. Products viewed effect
        print("\n2. PRODUCTS VIEWED EFFECT")
        print("-" * 80)
        
        conv_products = converters['unique_products_viewed']
        non_conv_products = non_converters['unique_products_viewed']
        
        u_stat, p_value = mannwhitneyu(conv_products, non_conv_products, alternative='two-sided')
        
        print(f"Mann-Whitney U test: U={u_stat:.4f}, p={p_value:.6f}")
        print(f"\nConverters:")
        print(f"  Mean: {conv_products.mean():.2f} products")
        print(f"  Median: {conv_products.median():.2f} products")
        print(f"\nNon-converters:")
        print(f"  Mean: {non_conv_products.mean():.2f} products")
        print(f"  Median: {non_conv_products.median():.2f} products")
        
        if p_value < 0.05:
            print("\n✓ Products viewed has SIGNIFICANT effect on conversion")
        
        results['products'] = {
            'u_stat': u_stat, 'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # 3. Total events effect
        print("\n3. TOTAL EVENTS EFFECT")
        print("-" * 80)
        
        conv_events = converters['total_events']
        non_conv_events = non_converters['total_events']
        
        u_stat, p_value = mannwhitneyu(conv_events, non_conv_events, alternative='two-sided')
        
        print(f"Mann-Whitney U test: U={u_stat:.4f}, p={p_value:.6f}")
        print(f"\nConverters median events: {conv_events.median():.0f}")
        print(f"Non-converters median events: {non_conv_events.median():.0f}")
        
        if p_value < 0.05:
            print("\nTotal events has SIGNIFICANT effect on conversion")
        
        results['events'] = {
            'u_stat': u_stat, 'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        self.test_results['engagement'] = results
        return results
    
    def test_correlations(self):
        """Test correlations between continuous variables."""
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        # Select continuous variables
        continuous_vars = [
            'session_duration_sec', 'unique_products_viewed', 
            'total_events', 'views', 'carts'
        ]
        
        # Test correlation with conversion
        print("\nCorrelation with Conversion:")
        print("-" * 80)
        
        results = {}
        
        for var in continuous_vars:
            if var in self.df_sessions.columns:
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(
                    self.df_sessions[var], 
                    self.df_sessions['converted']
                )
                
                # Spearman correlation (non-parametric)
                spearman_r, spearman_p = spearmanr(
                    self.df_sessions[var], 
                    self.df_sessions['converted']
                )
                
                print(f"\n{var}:")
                print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.6f})")
                print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.6f})")
                
                if pearson_p < 0.05:
                    if abs(pearson_r) < 0.3:
                        strength = "weak"
                    elif abs(pearson_r) < 0.7:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    
                    direction = "positive" if pearson_r > 0 else "negative"
                    print(f"  ✓ {strength.capitalize()} {direction} correlation (significant)")
                else:
                    print(f"  ✗ Not significant")
                
                results[var] = {
                    'pearson_r': pearson_r, 'pearson_p': pearson_p,
                    'spearman_r': spearman_r, 'spearman_p': spearman_p,
                    'significant': pearson_p < 0.05
                }
        
        self.test_results['correlations'] = results
        return results
    
    def generate_statistical_summary(self):
        """Generate comprehensive statistical summary report."""
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("=" * 80)
        
        summary = []
        
        # Temporal effects
        if 'temporal' in self.test_results:
            temporal = self.test_results['temporal']
            
            if temporal['hour_anova']['significant']:
                summary.append(
                    f"✓ Hour of day significantly affects conversion (F={temporal['hour_anova']['f_stat']:.2f}, "
                    f"p={temporal['hour_anova']['p_value']:.4f})"
                )
            
            if temporal['day_anova']['significant']:
                summary.append(
                    f"✓ Day of week significantly affects conversion (F={temporal['day_anova']['f_stat']:.2f}, "
                    f"p={temporal['day_anova']['p_value']:.4f})"
                )
            
            if temporal['weekend_ttest']['significant']:
                effect = self.interpret_cohens_d(temporal['weekend_ttest']['cohens_d'])
                summary.append(
                    f"✓ Weekend vs weekday shows significant difference (d={temporal['weekend_ttest']['cohens_d']:.2f}, "
                    f"{effect} effect)"
                )
        
        # User type effects
        if 'user_type' in self.test_results:
            user_type = self.test_results['user_type']
            
            if user_type['conversion']['significant']:
                effect = self.interpret_cohens_d(user_type['conversion']['cohens_d'])
                summary.append(
                    f"✓ Returning users convert significantly differently than new users "
                    f"(d={user_type['conversion']['cohens_d']:.2f}, {effect} effect)"
                )
        
        # Category effects
        if 'category' in self.test_results:
            category = self.test_results['category']
            
            if category['significant']:
                effect = self.interpret_cramers_v(category['cramers_v'])
                summary.append(
                    f"✓ Product category significantly influences conversion "
                    f"(Cramér's V={category['cramers_v']:.2f}, {effect} effect)"
                )
        
        # Engagement effects
        if 'engagement' in self.test_results:
            engagement = self.test_results['engagement']
            
            sig_factors = [k for k, v in engagement.items() if v['significant']]
            if sig_factors:
                summary.append(
                    f"Engagement metrics with significant effects: {', '.join(sig_factors)}"
                )
        
        # Correlations
        if 'correlations' in self.test_results:
            corr = self.test_results['correlations']
            
            sig_corr = [(k, v['pearson_r']) for k, v in corr.items() if v['significant']]
            if sig_corr:
                summary.append(
                    f"Variables significantly correlated with conversion: "
                    f"{', '.join([f'{k} (r={v:.2f})' for k, v in sig_corr])}"
                )
        
        print("\nKEY STATISTICAL FINDINGS:")
        print("-" * 80)
        for i, finding in enumerate(summary, 1):
            print(f"{i}. {finding}\n")
        
        return summary
    
    def run_full_analysis(self):
        """Run complete statistical analysis pipeline."""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "STATISTICAL ANALYSIS SUITE" + " " * 32 + "║")
        print("╚" + "=" * 78 + "╝")
        
        self.test_temporal_effects()
        self.test_user_type_effects()
        self.test_category_effects()
        self.test_engagement_effects()
        self.test_correlations()
        summary = self.generate_statistical_summary()
        
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS COMPLETE!")
        print("=" * 80)
        
        return self.test_results, summary
    
    def save_results(self, output_dir='../output'):
        """Save statistical test results."""
        import json
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results_serializable = convert_types(self.test_results)
        
        with open(f'{output_dir}/statistical_tests.json', 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        print(f"\n✓ Statistical results saved to {output_dir}/statistical_tests.json")


# Main execution
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║              E-COMMERCE STATISTICAL ANALYSIS SUITE                           ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Load session data
    print("Loading session data...")
    df_sessions = pd.read_csv('../output/session_features.csv')
    
    # Create analyzer
    analyzer = StatisticalAnalyzer(df_sessions)
    
    # Run full analysis
    test_results, summary = analyzer.run_full_analysis()
    
    # Save results
    analyzer.save_results()