from QuickEDA.plotting_manager import PlottingManager
from .stats import univariate_stats, split_univariate_stats, bivariate_stats, calculate_regression_stats, check_heteroscedasticity, calculate_group_stats, prepare_multivariate_data, calculate_vif, fit_linear_model, get_model_metrics, get_model_coefficients, stepwise_regression
# from .preprocessing import clean_data


class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.plotter = PlottingManager()

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

    def bivariate_analysis(self, label: str, plot_backend: str = 'seaborn'):
        """
        Perform comprehensive bivariate analysis between each feature and the target.
        
        Args:
            label: Target variable name for analysis
            plot_backend: Visualization library to use ('seaborn' or 'plotly')
            
        Returns:
            Tuple of:
            - DataFrame with statistical test results
            - Dictionary of visualization objects
            
        Examples:
            >>> analyzer = DataAnalyzer(df)
            >>> stats, plots = analyzer.bivariate_analysis('price')
            >>> stats, plots = analyzer.bivariate_analysis('price', plot_backend='plotly')
        """
        self.plotter.set_backend(plot_backend)
        results = {}
        plots = {}
        
        stats_df = bivariate_stats(self.df, label)
        
        for _, row in stats_df.iterrows():
            feature = row['feature']
            if row['test_type'] == 'pearson_r':
                # Generate scatter plot for numeric features
                stats = calculate_regression_stats(self.df[feature], self.df[label])
                het_test = check_heteroscedasticity(
                    self.df[[feature, label]], 
                    feature, 
                    label
                )
                plots[feature] = self.plotter.scatter(
                    self.df[feature], 
                    self.df[label],
                    stats=stats,
                    heteroscedasticity=het_test
                )
            elif row['test_type'] == 'anova_f':
                # Generate bar plot for categorical features
                stats = calculate_group_stats(self.df, feature, label)
                plots[feature] = self.plotter.bar_chart(
                    self.df, 
                    feature, 
                    label,
                    anova_results=stats['anova'],
                    pairwise_tests=stats['pairwise_tests']
                )
        
        return stats_df, plots

    def multivariate_analysis(self, target: str, method: str = 'full', min_features: int = 2):
        """
        Perform multivariate analysis on the dataset.
        
        Args:
            target: Name of the target variable
            method: Analysis type ('full', 'vif', or 'stepwise')
            min_features: Minimum features to keep (for stepwise)
            
        Returns:
            Analysis results (format varies by method)
            
        Examples:
            >>> # Full model with all features
            >>> results = analyzer.multivariate_analysis('price')
            
            >>> # Check multicollinearity
            >>> vif_results = analyzer.multivariate_analysis('price', method='vif')
            
            >>> # Stepwise feature selection
            >>> stepwise = analyzer.multivariate_analysis('price', method='stepwise', min_features=3)
        """
        prepared_df = prepare_multivariate_data(self.df)
        
        if method == 'vif':
            return calculate_vif(prepared_df, target)
        
        model = fit_linear_model(prepared_df, target)
        
        if method == 'full':
            return {
                'coefficients': get_model_coefficients(model),
                'metrics': get_model_metrics(model, prepared_df[target])
            }
        
        if method == 'stepwise':
            return stepwise_regression(self.df, target, min_features)
        
        raise ValueError("Method must be 'full', 'vif', or 'stepwise'")


    # def clean(self, strategies):
    #     self.df = clean_data(self.df, strategies)
    #     return self
