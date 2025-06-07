import pandas as pd
from typing import Union, Literal

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
