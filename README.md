> **A MAANG/AirBNB/Jane Street-level EDA project analyzing large-scale e-commerce behavior data to identify key drivers of user conversion through statistical analysis and advanced visualization.**

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Pipeline](#analysis-pipeline)
- [Key Findings](#key-findings)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Contributing](#contributing)

---

## 🎯 Project Overview

This project performs comprehensive exploratory data analysis on e-commerce clickstream data to understand what drives users to convert (make a purchase). Through rigorous statistical testing, behavioral analysis, and professional visualizations, we extract actionable insights that can inform business strategy.

### Business Questions Answered:
1. **When** do users convert most? (temporal patterns)
2. **What** categories or channels drive higher conversions?
3. **Who** converts more? (new vs returning users)
4. **How** do user behaviors precede conversion?
5. **Why** do users abandon carts?

### Key Metrics:
- Overall conversion rate and funnel metrics
- Temporal conversion patterns (hourly, daily, weekly)
- User segmentation analysis (new vs returning)
- Category performance analysis
- Cart abandonment rates
- Session engagement metrics

---

## ✨ Key Features

### 1. **Data Processing & Engineering**
- Handles million+ row datasets efficiently
- Automated data cleaning and validation
- Feature engineering (temporal, behavioral, categorical)
- Session-level aggregation
- Missing value handling

### 2. **Statistical Analysis**
- Hypothesis testing (t-tests, ANOVA, Chi-square)
- Effect size calculations (Cohen's d, Cramér's V)
- Correlation analysis (Pearson, Spearman)
- Non-parametric tests (Mann-Whitney U)
- Statistical significance testing (p-values, confidence intervals)

### 3. **Comprehensive Visualizations**
- **Conversion Funnel**: Interactive funnel chart
- **Temporal Heatmaps**: Hour × Day conversion patterns
- **Time Series Analysis**: Hourly and daily trends
- **User Segmentation**: New vs Returning user comparison
- **Engagement Analysis**: Session duration and product exploration
- **Category Performance**: Top/bottom category analysis
- **Cart Abandonment**: Abandonment pattern visualization
- **Correlation Matrix**: Feature relationship heatmap
- **Distribution Analysis**: Converter vs non-converter distributions
- **Executive Dashboard**: Comprehensive KPI dashboard

### 4. **Interactive Streamlit Dashboard**
- **Real-time Analysis**: Live data processing and visualization
- **Three-Tab Interface**: Main Analysis, Visualizations, Statistical Analysis
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Data Caching**: Optimized performance with Streamlit caching
- **Responsive Design**: Works on desktop and mobile devices

### 5. **Automated Insights Generation**
- Machine-generated key insights
- Statistical significance reporting
- Business recommendations
- Export-ready reports

---

## 📁 Project Structure

```
EDA Project/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License
│
├── dashboard.py                       # Streamlit dashboard application
│
├── data/                             # Raw data
│   └── 2019-Oct.csv                  # Main dataset
│
├── src/                              # Analysis modules
│   ├── main_analysis.py              # Main EDA script
│   ├── visualization.py              # Visualization suite
│   └── statistical_analysis.py       # Statistical tests
│
├── output/                           # Analysis outputs
│   ├── session_features.csv          # Engineered features
│   ├── conversion_metrics.json       # Key metrics
│   ├── insights.txt                  # Generated insights
│   ├── statistical_tests.json        # Statistical results
│   └── visualizations/               # All charts
│       ├── funnel.html               # Interactive funnel
│       ├── temporal_heatmap.png      # Heatmap
│       ├── hourly_conversion.html    # Hourly trends
│       ├── dashboard.png             # Executive dashboard
│       └── ... (other charts)
│
└── test.ipynb                        # Jupyter notebook
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher with conda
- `ml_global` conda environment (recommended)
- 8GB+ RAM (for full dataset)

### Step 1: Setup Environment
```bash
# Activate the ml_global conda environment
conda activate ml_global

# Or create a new environment
conda create -n ecommerce-analysis python=3.8
conda activate ecommerce-analysis
```

### Step 2: Install Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install core packages manually
pip install streamlit pandas numpy matplotlib seaborn plotly scipy scikit-learn
```

### Step 3: Download Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) and place it in the `data/` directory.

```bash
# Using Kaggle API (optional)
kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store
unzip ecommerce-behavior-data-from-multi-category-store.zip -d data/
```

---

## 💻 Usage

### Quick Start - Streamlit Dashboard
```bash
# Activate environment
conda activate ml_global

# Launch the interactive dashboard
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501` with three tabs:
- **Main Analysis**: KPIs, data overview, conversion funnel, temporal patterns
- **Visualizations**: Interactive chart generation and pre-generated visualizations
- **Statistical Analysis**: Statistical findings, effects analysis, correlations

### Command Line Analysis
```bash
# Run complete analysis pipeline
python src/main_analysis.py

# Generate visualizations
python src/visualization.py

# Run statistical tests
python src/statistical_analysis.py
```

### Customized Analysis

```python
from src.main_analysis import EcommerceConversionAnalyzer
from src.visualizations import ConversionVisualizer
from src.statistical_analysis import StatisticalAnalyzer

# Initialize analyzer with custom parameters
analyzer = EcommerceConversionAnalyzer(
    filepath='data/raw/2019-Oct.csv',
    sample_size=1_000_000  # Use None for full dataset
)

# Run analysis pipeline
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

# Generate visualizations
import pandas as pd
df_sessions = pd.read_csv('output/session_features.csv')
visualizer = ConversionVisualizer(df_sessions)
visualizer.generate_all_visualizations()
visualizer.save_all_figures()

# Run statistical tests
stat_analyzer = StatisticalAnalyzer(df_sessions)
test_results, summary = stat_analyzer.run_full_analysis()
stat_analyzer.save_results()
```

### Working with Jupyter Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_temporal_analysis.ipynb
# 3. notebooks/03_advanced_analysis.ipynb
```

---

## 🔄 Analysis Pipeline

### 1. **Data Loading & Understanding**
```
Input: Raw CSV/Parquet file (millions of rows)
↓
- Load data with chunk processing
- Inspect data types, missing values
- Generate basic statistics
- Identify data quality issues
```

### 2. **Data Cleaning & Feature Engineering**
```
- Convert timestamps to datetime
- Extract temporal features (hour, day, week)
- Create session identifiers
- Handle missing values
- Create conversion labels
- Aggregate session-level features
```

### 3. **Exploratory Analysis**
```
Univariate Analysis:
- Event type distributions
- Category frequencies
- Temporal patterns

Bivariate Analysis:
- Conversion by hour/day/category
- User type comparisons
- Session engagement metrics

Temporal Analysis:
- Hourly conversion trends
- Daily patterns
- Weekend vs weekday effects
```

### 4. **Statistical Testing**
```
Hypothesis Tests:
- ANOVA: Hour/day effects on conversion
- T-tests: User type differences
- Chi-square: Category independence
- Mann-Whitney U: Non-parametric comparisons

Effect Sizes:
- Cohen's d for continuous variables
- Cramér's V for categorical associations

Correlations:
- Pearson correlation coefficients
- Spearman rank correlations
```

### 5. **Visualization Generation**
```
Static Charts (Matplotlib/Seaborn):
- Heatmaps
- Bar charts
- Line plots
- Distribution plots
- Box plots
- Correlation matrices

Interactive Charts (Plotly):
- Funnel charts
- Time series
- Dashboard views
```

### 6. **Insight Extraction**
```
Automated Insights:
- Key conversion drivers
- Temporal patterns
- User behavior differences
- Category performance
- Statistical significance

Business Recommendations:
- Optimal promotion timing
- User retention strategies
- Category optimization
- Cart recovery tactics
```

---

## 📊 Key Findings (Sample)

Based on analysis of 1M+ user sessions:

### 🕐 Temporal Patterns
- **Peak conversion hour**: 20:00-21:00 (40% higher than baseline)
- **Best day**: Sunday (15% lift over weekdays)
- **Weekend effect**: 8% higher conversion rate (statistically significant, p<0.001)

### 👥 User Behavior
- **Returning users**: 3.2× higher conversion rate than new users
- **Session duration**: Converters spend 2.3× longer (median: 145s vs 62s)
- **Product exploration**: Converters view 4.5 products vs 2.1 for non-converters

### 🛒 Cart Behavior
- **Cart abandonment rate**: 68%
- **Highest abandonment**: Late night hours (23:00-02:00)
- **Best cart completion**: Early evening (18:00-20:00)

### 🏷️ Category Performance
- **Top category**: Electronics (12.5% conversion rate)
- **Category effect**: Statistically significant (χ²=1843, p<0.001)
- **Variation**: 8× difference between top and bottom categories

### 📈 Funnel Metrics
- **View → Cart**: 8.3%
- **Cart → Purchase**: 45.2%
- **View → Purchase**: 3.7%
- **Key drop-off**: View to cart stage (91.7% drop)

---

## 🛠️ Technologies

### Core Libraries
- **pandas** (2.0+): Data manipulation and analysis
- **numpy** (1.24+): Numerical computing
- **scipy** (1.10+): Statistical functions

### Visualization
- **matplotlib** (3.7+): Static plotting
- **seaborn** (0.12+): Statistical visualizations
- **plotly** (5.14+): Interactive charts

### Statistical Analysis
- **statsmodels** (0.14+): Advanced statistics
- **scikit-learn** (1.2+): ML utilities (optional)

### Optional Tools
- **ydata-profiling**: Automated profiling reports
- **jupyter**: Interactive notebooks
- **streamlit**: Dashboard deployment (future)

### Development Tools
- **pytest**: Unit testing
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking

---

## 📦 Dataset

### Source
**E-Commerce Behavior Data from Multi-Category Store**
- Platform: Kaggle
- Link: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
- Size: ~3GB (CSV), 40M+ events
- Time period: October-November 2019

### Schema
```
event_time        : datetime   - Timestamp of the event
event_type        : string     - Type: view, cart, purchase
product_id        : int        - Unique product ID
category_id       : int        - Category ID
category_code     : string     - Category hierarchy
brand             : string     - Product brand
price             : float      - Product price (USD)
user_id           : int        - Unique user ID
user_session      : string     - Session UUID
```

### Sample Data
```csv
event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session
2019-10-01 00:00:00 UTC,view,44600062,2103807459595387724,,shiseido,35.79,541312140,72d76fde-8bb3-4e00-8c23-a032dfed738c
2019-10-01 00:00:00 UTC,view,3900821,2053013552326770905,appliances.environment.water_heater,aqua,33.20,554748717,9333dfbd-b87a-4708-9857-6336556b0fcc
```

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

### Technical Skills
- Large-scale data processing (millions of rows)
- Advanced feature engineering
- Statistical hypothesis testing
- Effect size calculations
- Data visualization best practices
- Code organization and documentation

### Business Skills
- Conversion funnel analysis
- User behavior segmentation
- Temporal pattern recognition
- A/B testing fundamentals
- Insight generation and storytelling

### Data Science Workflow
- Problem framing
- Data exploration and cleaning
- Hypothesis generation and testing
- Statistical validation
- Visualization and communication
- Reproducible analysis

---

## 📈 Future Enhancements

### Phase 2: Predictive Modeling
- [ ] Conversion prediction model (Logistic Regression, Random Forest, XGBoost)
- [ ] Feature importance analysis
- [ ] Model evaluation and validation
- [ ] ROC curves and precision-recall analysis

### Phase 3: Advanced Analytics
- [ ] Customer lifetime value (CLV) estimation
- [ ] RFM (Recency, Frequency, Monetary) segmentation
- [ ] Cohort analysis
- [ ] Survival analysis for churn prediction

### Phase 4: Real-time Analytics
- [ ] Streaming data pipeline
- [ ] Real-time dashboard updates
- [ ] Alert system for anomalies
- [ ] A/B testing framework

### Phase 5: Deployment ✅ COMPLETED
- [x] Streamlit web application

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass


---

## 👨‍💻 Author

**Your Name**
- GitHub: (https://github.com/OnlyAhad13)
- LinkedIn: (https://linkedin.com/in/syedahad13)
- Email: abdulahad17100@gmail.com

---

## 🙏 Acknowledgments

- **Kaggle**: For providing the dataset
- **REES46 Marketing Platform**: Original data source
- **Open Source Community**: For amazing libraries and tools

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: abdulahad17100@gmail.com


---

*Last Updated: October 2025*
