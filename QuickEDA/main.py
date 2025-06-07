from .stats import univariate_stats, split_univariate_stats
# from .preprocessing import clean_data


class DataAnalyzer:
    def __init__(self, df, backend="plotly"):
        self.df = df
        self.set_backend(backend)

    def set_backend(self, backend):
        self.backend = backend

    def univariate_analysis(self, split_results=False, sort_by="skew"):
            """
            Perform univariate analysis.
            
            Parameters:
            -----------
            split_results : bool (default=False)
                If True, returns separated numeric/categorical results
            sort_by : str (default="skew")
                Column to sort results by
            """
            if split_results:
                return split_univariate_stats(self.df)
            return univariate_stats(self.df, sort_by)

    # def clean(self, strategies):
    #     self.df = clean_data(self.df, strategies)
    #     return self
