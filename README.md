# QuickEDA

[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automated Exploratory Data Analysis (EDA) toolkit that delivers statistical insights and publication-ready visualizations with 3 lines of code.

```python
from auto_eda import DataAnalyzer

analyzer = DataAnalyzer(df)
analyzer.explore_all()  # Generates complete report
```

## 🔍 Features

- **Smart Analysis**  
  Univariate, bivariate, and multivariate diagnostics
- **Visualization Backends**  
  Switch between Seaborn or Plotly with one parameter
- **Statistical Rigor**  
  Built-in hypothesis testing and model diagnostics
- **Zero Configuration**  
  Works out-of-the-box while allowing customization

## ⚡ Quick Start

### Installation
```bash
pip install quickeda
```

### Basic Usage
```python
# Initialize with your dataframe
analyzer = DataAnalyzer(df)

# 1. Univariate stats (all features)
stats = analyzer.univariate_analysis(sort_by="skew")

# 2. Bivariate analysis (against target)
plots = analyzer.bivariate_analysis('price', plot_backend='plotly')

# 3. Check multicollinearity
vif_results = analyzer.multivariate_analysis('price', method='vif')
```

## 📊 Visualization Examples

### Switch Backends Seamlessly
```python
analyzer.plotter.set_backend('seaborn')  # Default
analyzer.plotter.set_backend('plotly')   # Interactive
```

### Available Plots
| Plot Type | Description |
|-----------|-------------|
| `.scatter()` | Regression plots with stats |
| `.bar_chart()` | Group comparisons with ANOVA |
| `.histogram()` | Distribution analysis |

## 📈 Advanced Usage

### Stepwise Feature Selection
```python
stepwise = analyzer.multivariate_analysis(
    target='price',
    method='stepwise',
    min_features=3
)
```

### Access Raw Statistical Functions
```python
from auto_eda.stats import calculate_vif
vif = calculate_vif(df, target='price')
```

## 🤝 Contributing
PRs are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License
MIT © 2023 Shubham Patel
