"""
Takes a collection of DataFrames and performs summary statistics on them.

Classes
-------
Summary
    Performs summary statistics on provided dataframes

"""
from typing import Literal, Union

import pandas as pd


class Summary:
    """ Generates summary statistics for provided dataframes

    Attributes
    ----------
    _dataframes : dict[str, pd.DataFrame]
        Dataframes to perform summary stats on

    group :pd.GroupBy
        Dataframes grouped by index

    """
    def __init__(self, dataframes: dict[str, pd.DataFrame]):
        """
        Initialises the class and groups the dataframe by the index

        Parameters
        ----------
        dataframes : dict[str, pd.DataFrame]
            Dictionary of dataframes to summarise
        """
        self._dataframes = dataframes
        dfs = list(self._dataframes.values())
        for df in dfs:
            scores_to_invert = list(
                    filter(
                        lambda x: x in df.columns,
                        [
                            "Explained Variance Score",
                            "r2",
                            "Mean Pinball Deviance",
                            "Mean Poisson Deviance",
                            "Mean Gamma Deviance",
                            ]
                        )
                )
            if scores_to_invert:
                df.loc[:, scores_to_invert] = 1 - df.loc[:, scores_to_invert]

        self.group = pd.concat(dfs).groupby(by=dfs[0].index.name, level=0)

    def mean(self) -> pd.DataFrame:
        """
        Get the mean of the groupby object
        """
        return self.group.mean()

    def median(self) -> pd.DataFrame:
        """
        Get the median of the groupby object
        """
        return self.group.median()

    def max(self) -> pd.DataFrame:
        """
        Get the max of the groupby object
        """
        return self.group.max()

    def min(self) -> pd.DataFrame:
        """
        Get the min of the groupby object
        """
        return self.group.min()

    def diff_from_mean(self) -> dict[str, pd.DataFrame]:
        """
        Get difference from the mean of the groupby object
        """
        diff_dict: dict[str, pd.DataFrame] = dict()
        mean_df = self.mean()
        for key, df in self._dataframes.items():
            diff_dict[key] = df - mean_df
        return diff_dict

    def best_performing(
            self,
            summate: Literal["all", "key", "row"] = "key"
            ) -> Union[dict[str, int], dict[str, dict[str, int]]]:
        if summate == "all":
            return self._best_performing_all()
        elif summate == "key":
            return self._best_performing_key()
        elif summate == "row":
            return self._best_performing_row()

    def _best_performing_all(self) -> dict[str, dict[str, int]]:
        count_dict: dict[str, dict[str, int]] = dict()
        min_df = self.min()
        for key, df in self._dataframes.items():
            counts = df.eq(min_df).sum(axis=1)
            count_dict[key] = counts.to_dict()
        return count_dict

    def _best_performing_row(self) -> dict[str, int]:
        all_counts = self._best_performing_all()
        count_series = pd.Series(list(all_counts.values())[0])
        for counts in list(all_counts.values())[1:]:
            count_series = count_series + pd.Series(counts)
        count_dict: dict[str, int] = count_series.to_dict()
        return count_dict

    def _best_performing_key(self) -> dict[str, int]:
        all_counts = self._best_performing_all()
        count_series = pd.Series()
        for key, counts in all_counts.items():
            count_series[key] = pd.Series(counts).sum()
        count_dict: dict[str, int] = count_series.to_dict()
        return count_dict
