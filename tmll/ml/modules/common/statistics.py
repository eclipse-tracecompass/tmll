from typing import Literal
import pandas as pd
from scipy.stats import normaltest


class Statistics:

    @staticmethod
    def get_correlation_method(series1: pd.Series, series2: pd.Series) -> Literal['pearson', 'kendall', 'spearman']:
        """
        Determine the most appropriate correlation method based on data distribution.
        It uses D'Agostino's test to determine if the data is normally distributed.

        :param series1: First series to compare
        :type series1: pd.Series
        :param series2: Second series to compare
        :type series2: pd.Series
        :return: Correlation method to use
        :rtype: Literal['pearson', 'kendall', 'spearman']
        """
        _, p1 = normaltest(series1)
        _, p2 = normaltest(series2)

        # If both series are normally distributed (p > 0.05), use Pearson
        if p1 > 0.05 and p2 > 0.05:
            return 'pearson'
        # If data is ordinal or highly non-normal, use Kendall
        elif (series1.nunique() < 10 or series2.nunique() < 10):
            return 'kendall'
        # Otherwise, use Spearman
        else:
            return 'spearman'

    @staticmethod
    def get_coefficient_of_variation(series: pd.Series, percentile: float = 0.99) -> float:
        """
        Calculate the coefficient of variation for a given series.

        :param series: Series to calculate coefficient of variation for
        :type series: pd.Series
        :param percentile: Percentile to use for calculation
        :type percentile: float
        :return: Coefficient of variation
        :rtype: float
        """
        if series.empty:
            return -1

        data = series[series < series.quantile(percentile)]

        mean = data.mean()
        std = data.std()
        cv = std / mean if mean > 0 else float("inf")

        if cv == float("inf"):
            return -1

        return cv
