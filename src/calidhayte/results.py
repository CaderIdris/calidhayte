from collections import defaultdict
from pathlib import Path
import sqlite3 as sql
from typing import Literal, Optional

import pandas as pd
from sklearn import metrics as met

from .calibrate import Calibrate
from .prepare import prepare_datasets


class Results:
    """Calculates errors between "true" and "predicted" measurements, plots
    graphs and returns all results

    ```

    Attributes
    ----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    coefficients : pd.DataFrame
        Calibration coefficients
    _errors : dict[str, pd.DataFrame]
        Dictionary of dataframes, each key representing a
        different subset of the data
    _cal : Calibrate
        Calibrate object that contains train and test calibrated by
        coefficients
    y_pred : dict
        Calibrated x measurements
    x_name : str
        Name of x device
    y_name : str
        Name of y device

    Methods
    -------
    explained_variance_score()
        Calculate the explained variance score
        between the true (y) measurements and all predicted (x) measurements
    max()
        Calculate the max error between the true (y) measurements and all
        predicted (x) measurements
    mean_absolute()
        Calculate the mean absolute error between the true (y)
        measurements and all predicted (x) measurements
    root_mean_squared()
        Calculate the root mean squared error between the
        true (y) measurements and all predicted (x) measurements
    root_mean_squared_log()
        Calculate the root_mean_squared_log error
        between the true (y) measurements and all predicted (x) measurements
    median_absolute()
        Calculate the median absolute error between the true
        (y) measurements and all predicted (x) measurements
    mean_absolute_percentage()
        Calculate the mean absolute percentage error
        between the true (y) measurements and all predicted (x) measurements
    r2()
        Calculate the r2 score between the true (y) measurements and all
        predicted (x) measurements
    mean_poisson_deviance()
        Calculate the mean poisson deviance between the
        true (y) measurements and all predicted (x) measurements
    mean_gamma_deviance()
        Calculate the mean gamma deviance between the true
        (y) measurements and all predicted (x) measurements
    mean_tweedie_deviance()
        Calculate the mean tweedie deviance between the
        true (y) measurements and all predicted (x) measurements
    mean_pinball_loss()
        Calculate the mean pinball loss between the true
        (y) measurements and all predicted (x) measurements
    return_errors()
        Returns dictionary of all recorded errors
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        coefficients: pd.DataFrame,
        datasets_to_use: list[
            Literal[
                "Calibrated Train",
                "Calibrated Test",
                "Calibrated Full",
                "Uncalibrated Train",
                "Uncalibrated Test",
                "Uncalibrated Full",
                "Minimum",
                "Maximum"
                ]
            ] = [
            "Calibrated Test",
            "Uncalibrated Test"
            ],
        x_name: Optional[str] = None,
        y_name: Optional[str] = None
    ):
        """Initialise the class

        Parameters
        ----------
        train : pd.DataFrame
            Training data
        test : pd.DataFrame
            Testing data
        coefficients : pd.DataFrame
            Calibration coefficients
        x_name : str, optional
            Name of device that was calibrated
            (Default is None)
        y_name : str, optional
            Name of ground truth device
            (Default is None)
        """
        self.train = train
        self.test = test
        self.coefficients = coefficients
        self._errors: defaultdict[str, pd.DataFrame] = defaultdict(
                pd.DataFrame
                )
        self._cal = Calibrate(
                self.train,
                self.test,
                self.coefficients
                )
        self.y_subsets = self._cal.return_measurements()
        self.y_full = self._cal.join_measurements()
        self._datasets = prepare_datasets(
                datasets_to_use,
                self.train,
                self.test,
                self.y_full,
                self.y_subsets
                )
        self.x_name = x_name
        self.y_name = y_name

    def explained_variance_score(self):
        """Calculate the explained variance score between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.explained_variance_score
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Explained Variance Score"
                        ] = met.explained_variance_score(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def max(self):
        """Calculate the max error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.max_error
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Max Error"
                        ] = met.max_error(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def mean_absolute(self):
        """Calculate the mean absolute error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_absolute_error
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Mean Absolute Error"
                        ] = met.mean_absolute_error(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def root_mean_squared(self):
        """Calculate the root mean squared error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_squared_error
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Root Mean Squared Error"
                        ] = met.mean_squared_error(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var],
                                squared=False
                                )

    def root_mean_squared_log(self):
        """Calculate the root mean squared log error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_squared_log_error
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Root Mean Squared Log Error"
                        ] = met.mean_squared_log_error(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var],
                                squared=False
                                )

    def median_absolute(self):
        """Calculate the median absolute error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.median_absolute_error
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Median Absolute Error"
                        ] = met.median_absolute_error(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def mean_absolute_percentage(self):
        """Calculate the mean absolute percentage error between the true
        values (y) and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_absolute_percentage_error
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Mean Absolute Percentage Error"
                        ] = met.mean_absolute_percentage_error(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def r2(self):
        """Calculate the r2 between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.r2_score
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "r2"
                        ] = met.r2_score(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def mean_poisson_deviance(self):
        """Calculate the mean poisson deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_poisson_deviance
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Mean Poisson Deviance"
                        ] = met.mean_poisson_deviance(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def mean_gamma_deviance(self):
        """Calculate the mean gamma deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_gamma_deviance
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Mean Gamma Deviance"
                        ] = met.mean_gamma_deviance(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def mean_tweedie_deviance(self):
        """Calculate the mean tweedie deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_tweedie_deviance
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Mean Tweedie Deviance"
                        ] = met.mean_tweedie_deviance(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def mean_pinball_loss(self):
        """Calculate the mean pinball loss between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_pinball_loss
        """
        for dset_key, dset in self._datasets.items():
            for var in dset['x'].columns:
                self._errors[dset_key].loc[
                        var, "Mean Pinball Deviance"
                        ] = met.mean_pinball_loss(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def return_errors(self) -> dict[str, pd.DataFrame]:
        """Returns all calculated errors in dataframe format"""
        return dict(self._errors)

    def save_results(self, path: str):
        for key, item in self._errors.items():
            self._errors[key] = pd.DataFrame(data=dict(item))
            if "Variable" in self._errors[key].columns:
                self._errors[key] = self._errors[key].set_index("Variable")
                vars_list = self._errors[key].columns.to_list()
                for vars in vars_list:
                    error_results = pd.DataFrame(self._errors[key][vars])
                    coefficients = pd.DataFrame(self.coefficients.loc[vars].T)
                    directory = Path(f"{path}/{key}/{vars}")
                    directory.mkdir(parents=True, exist_ok=True)
                    con = sql.connect(f"{directory.as_posix()}/Results.db")
                    error_results.to_sql(
                        name="Errors",
                        con=con,
                        if_exists="replace",
                        index=True
                    )
                    coefficients.to_sql(
                        name="Coefficients",
                        con=con,
                        if_exists="replace",
                        index=True
                    )
                    con.close()
