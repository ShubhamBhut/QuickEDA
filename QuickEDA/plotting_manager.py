class PlottingManager:
    """Ultra-simple plotting backend switcher"""
    
    def __init__(self):
        self.backend = 'seaborn'  # default
    
    def set_backend(self, backend_name):
        """Set backend (seaborn|plotly)"""
        if backend_name not in ['seaborn', 'plotly']:
            raise ValueError("Only 'seaborn' or 'plotly' supported")
        self.backend = backend_name
    
    def scatter(self, x, y, **kwargs):
        """Create scatter plot with current backend"""
        if self.backend == 'seaborn':
            import seaborn as sns
            return sns.jointplot(x=x, y=y, kind='reg', **kwargs)
        else:
            import plotly.express as px
            return px.scatter(x=x, y=y, trendline="ols", **kwargs)
    
    def bar_chart(self, x, y, **kwargs):
        """Create bar chart with current backend"""
        if self.backend == 'seaborn':
            import seaborn as sns
            return sns.barplot(x=x, y=y, **kwargs)
        else:
            import plotly.express as px
            return px.bar(x=x, y=y, **kwargs)
