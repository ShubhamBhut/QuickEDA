from .stats import univariate_stats, split_univariate_stats
# from .preprocessing import clean_data


class DataAnalyzer:
    def __init__(self, df, backend="plotly"):
        self.df = df
        self.set_backend(backend)

    def set_backend(self, backend):
        self.backend = backend

    def univariate_analysis(self, split_results: bool = False, sort_by: str = "skew"):
        """
        Perform complete univariate analysis on the dataset.
        
        Parameters:
        -----------
        split_results : bool, optional (default=False)
            If True, returns results separated into numeric and categorical columns
        sort_by : str, optional (default="skew")
            Metric to sort results by (options: 'skew', 'kurt', 'missing', etc.)
            
        Returns:
        --------
        Union[pd.DataFrame, dict]
            - If split_results=False: Single DataFrame with all stats
            - If split_results=True: Dictionary with keys 'numeric' and 'categorical'
            
        Examples:
        --------
        >>> analyzer = DataAnalyzer(df)
        >>> # Get combined results
        >>> combined = analyzer.univariate_analysis()
        >>> # Get split results
        >>> split = analyzer.univariate_analysis(split_results=True)
        """
        if split_results:
            return split_univariate_stats(self.df)
        return univariate_stats(self.df, sort_by)


    # def clean(self, strategies):
    #     self.df = clean_data(self.df, strategies)
    #     return self
