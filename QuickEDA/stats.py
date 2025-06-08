from typing import Union, Literal
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import statsmodels.api as sm
from statsmodels.formula.api import ols

def univariate_stats(
    df: pd.DataFrame,
    sort_by: Union[Literal["skew", "kurt", "missing", "unique", "std"], str] = "skew"
) -> pd.DataFrame:
    """
    Calculate comprehensive univariate statistics for all columns in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to analyze
    sort_by : str, optional (default="skew")
        Column to sort the results by (options: 'skew', 'kurt', 'missing', 'unique', 'std')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe containing univariate statistics for each column
    """
    STATS_COLUMNS = [
        'count', 'missing', 'unique', 'dtype', 
        'min', '25%', 'median', '75%', 'max', 
        'mean', 'mode', 'std', 'skew', 'kurt', 
        'numeric'
    ]
    
    output_df = pd.DataFrame(columns=STATS_COLUMNS)
    
    for col in df.columns:
        col_series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(col_series)
        
        stats = {
            'count': col_series.count(),
            'missing': col_series.isnull().sum(),
            'unique': col_series.nunique(),
            'dtype': col_series.dtype,
            'numeric': is_numeric
        }
        
        if is_numeric:
            stats.update({
                'min': col_series.min(),
                '25%': col_series.quantile(0.25),
                'median': col_series.median(),
                '75%': col_series.quantile(0.75),
                'max': col_series.max(),
                'mean': col_series.mean(),
                'std': col_series.std(),
                'skew': col_series.skew(),
                'kurt': col_series.kurt(),
                'mode': col_series.mode().values[0]
            })
        else:
            stats.update({
                'min': '-', '25%': '-', 'median': '-', '75%': '-', 'max': '-',
                'mean': '-', 'std': '-', 'skew': '-', 'kurt': '-',
                'mode': col_series.mode().values[0]
            })
        
        output_df.loc[col] = stats
    
    return output_df.sort_values(
        by=["numeric", sort_by], 
        ascending=[False, sort_by != "missing"]
    )


def split_univariate_stats(df):
    stats = univariate_stats(df)
    return {
        'numeric': stats[stats['numeric']],
        'categorical': stats[~stats['numeric']]
    }


def check_heteroscedasticity(df: pd.DataFrame, feature: str, label: str) -> pd.DataFrame:
    """
    Check for heteroscedasticity using Breusch-Pagan and White tests.
    
    Args:
        df: Input DataFrame
        feature: Independent variable name
        label: Dependent variable name
        
    Returns:
        DataFrame with test results
    """
    model = ols(formula=f"{label} ~ {feature}", data=df).fit()
    output_df = pd.DataFrame(columns=['LM stat', 'LM p-value', 'F-stat', 'F p-value'])
    
    # Breusch-Pagan test
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    output_df.loc['Breusch-Pagan'] = bp_test
    
    # White test (may fail in some cases)
    try:
        white_test = het_white(model.resid, model.model.exog)
        output_df.loc['White'] = white_test
    except Exception as e:
        print(f"Unable to calculate White test: {str(e)}")
    
    return output_df.round(3)

def calculate_regression_stats(feature: pd.Series, label: pd.Series) -> dict:
    """
    Calculate regression statistics between two numeric variables.
    
    Args:
        feature: Independent variable
        label: Dependent variable
        
    Returns:
        Dictionary of regression statistics
    """
    m, b, r, p, _ = stats.linregress(feature, label)
    return {
        'slope': round(m, 3),
        'intercept': round(b, 3),
        'r_squared': round(r**2, 3),
        'p_value': round(p, 3),
        'feature_skew': round(feature.skew(), 3),
        'label_skew': round(label.skew(), 3)
    }

def calculate_group_stats(df: pd.DataFrame, feature: str, label: str) -> dict:
    """
    Calculate ANOVA and pairwise t-tests between groups.
    
    Args:
        df: Input DataFrame
        feature: Categorical grouping variable
        label: Numeric target variable
        
    Returns:
        Dictionary containing ANOVA and pairwise test results
    """
    groups = df[feature].unique()
    df_grouped = df.groupby(feature)
    group_data = [df_grouped.get_group(g)[label] for g in groups]
    
    # ANOVA
    f_stat, p_value = stats.f_oneway(*group_data)
    
    # Pairwise t-tests
    ttests = []
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if j > i:
                data1 = df[df[feature] == group1][label]
                data2 = df[df[feature] == group2][label]
                
                if len(data1) < 2 or len(data2) < 2:
                    print(f"{group1} (n={len(data1)}) vs {group2} (n={len(data2)}): Not enough samples")
                else:
                    t, p = stats.ttest_ind(data1, data2)
                    ttests.append({
                        'group1': group1,
                        'group2': group2,
                        't_stat': round(t, 3),
                        'p_value': round(p, 3),
                        'significant': p < (0.05/len(ttests)) if ttests else p < 0.05
                    })
    
    return {
        'anova': {'f_stat': round(f_stat, 3), 'p_value': round(p_value, 3)},
        'pairwise_tests': ttests,
        'bonferroni_threshold': 0.05/len(ttests) if ttests else 0.05
    }

def bivariate_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Calculate bivariate statistics between each feature and the target label.
    
    Args:
        df: Input DataFrame
        label: Target variable name
        
    Returns:
        DataFrame with statistical test results for each feature
    """
    results = []
    
    for col in df.columns:
        if col == label:
            continue
            
        if df[col].isnull().sum() > 0:
            results.append({
                'feature': col,
                'test_type': None,
                'statistic': None,
                'p_value': "nulls"
            })
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            # Pearson correlation for numeric features
            r, p = stats.pearsonr(df[label], df[col])
            results.append({
                'feature': col,
                'test_type': 'pearson_r',
                'statistic': round(r, 3),
                'p_value': round(p, 3)
            })
        else:
            # ANOVA for categorical features
            f_stat, p = calculate_group_stats(df[[col, label]], col, label)['anova'].values()
            results.append({
                'feature': col,
                'test_type': 'anova_f',
                'statistic': round(f_stat, 3),
                'p_value': round(p, 3)
            })
    
    output_df = pd.DataFrame(results)
    return output_df.sort_values(
        by=['statistic'], 
        key=abs, 
        ascending=False
    ).reset_index(drop=True)


