from collections import defaultdict
import re
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics as met

mpl.use("pgf")  # Used to make pgf files for latex
from .calibrate import Calibrate

plt.rcParams.update({"figure.max_open_warning": 0})


class Results:
    """Calculates errors between "true" and "predicted" measurements, plots
    graphs and returns all results

    Attributes:
        train (DataFrame): Training data

        test (DataFrame): Testing data

        coefficients (DataFrame): Calibration coefficients

        _errors (dict): Dictionary of dataframes, each key representing a
        different calibration method

        y_pred (dict): Calibrated x measurements

        combos (list): List of all possible variable and dataset combos

        x_name (str): Name of x device

        y_name (str): Name of y device

    Methods:
        _calibrate: Calibrate all x measurements with provided coefficients.
        This function splits calibrations depending on whether the coefficients
        were derived using skl or pymc

        _pymc_calibrate: Calibrates x measurements with provided pymc
        coefficients. Returns mean, max and min calibrations.

        _skl_calibrate: Calibrates x measurements with provided skl
        coefficients.

        _get_all_combos: Return all possible combinations of datasets (e.g
        calibrated test, uncalibrated train) for every method

        _all_combos: Return all possible combinations of datasets (e.g
        calibrated test, uncalibrated train) for single method

        explained_variance_score: Calculate the explained variance score
        between the true (y) measurements and all predicted (x) measurements

        max: Calculate the max error between the true (y) measurements and all
        predicted (x) measurements

        mean_absolute: Calculate the mean absolute error between the true (y)
        measurements and all predicted (x) measurements

        root_mean_squared: Calculate the root mean squared error between the
        true (y) measurements and all predicted (x) measurements

        root_mean_squared_log: Calculate the root_mean_squared_log error
        between the true (y) measurements and all predicted (x) measurements

        median_absolute: Calculate the median absolute error between the true
        (y) measurements and all predicted (x) measurements

        mean_absolute_percentage: Calculate the mean absolute percentage error
        between the true (y) measurements and all predicted (x) measurements

        r2: Calculate the r2 score between the true (y) measurements and all
        predicted (x) measurements

        mean_poisson_deviance: Calculate the mean poisson deviance between the
        true (y) measurements and all predicted (x) measurements

        mean_gamma_deviance: Calculate the mean gamma deviance between the true
        (y) measurements and all predicted (x) measurements

        mean_tweedie_deviance: Calculate the mean tweedie deviance between the
        true (y) measurements and all predicted (x) measurements

        mean_pinball_loss: Calculate the mean pinball loss between the true
        (y) measurements and all predicted (x) measurements

        return_errors: Returns dictionary of all recorded errors
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        coefficients: pd.DataFrame,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None
    ):
        """Initialise the class

        Keyword Arguments:
            train (DataFrame): Training data

            test (DataFrame): Testing data

            coefficients (DataFrame): Calibration coefficients

            comparison_name (String): Name of the comparison
        """
        self.train = train
        self.test = test
        self.coefficients = coefficients
        self._errors: dict[str, pd.DataFrame] = dict()
        self._cal = Calibrate(
                self.train,
                self.test,
                self.coefficients
                )
        self.y_pred = self._cal.return_measurements()
        self.x_name = x_name
        self.y_name = y_name

    def explained_variance_score(self):
        """Calculate the explained variance score between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.explained_variance_score
        """
        error_name = "Explained Variance Score"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(
                    met.explained_variance_score(true, pred)
                )

    def max(self):
        """Calculate the max error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.max_error
        """
        error_name = "Max Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(met.max_error(true, pred))

    def mean_absolute(self):
        """Calculate the mean absolute error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_absolute_error
        """
        error_name = "Mean Absolute Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(met.mean_absolute_error(true, pred))

    def root_mean_squared(self):
        """Calculate the root mean squared error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_squared_error
        """
        error_name = "Root Mean Squared Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(
                    met.mean_squared_error(true, pred, squared=False)
                )

    def root_mean_squared_log(self):
        """Calculate the root mean squared log error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_squared_log_error
        """
        error_name = "Root Mean Squared Log Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(
                    met.mean_squared_log_error(true, pred, squared=False)
                )

    def median_absolute(self):
        """Calculate the median absolute error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.median_absolute_error
        """
        error_name = "Median Absolute Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(met.median_absolute_error(true, pred))

    def mean_absolute_percentage(self):
        """Calculate the mean absolute percentage error between the true
        values (y) and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_absolute_percentage_error
        """
        error_name = "Mean Absolute Percentage Error"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(
                    met.mean_absolute_percentage_error(true, pred)
                )

    def r2(self):
        """Calculate the r2 between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.r2_score
        """
        error_name = "r2"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(met.r2_score(true, pred))

    def mean_poisson_deviance(self):
        """Calculate the mean poisson deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_poisson_deviance
        """
        error_name = "Mean Poisson Deviance"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(met.mean_poisson_deviance(true, pred))

    def mean_gamma_deviance(self):
        """Calculate the mean gamma deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_gamma_deviance
        """
        error_name = "Mean Gamma Deviance"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(met.mean_gamma_deviance(true, pred))

    def mean_tweedie_deviance(self):
        """Calculate the mean tweedie deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_tweedie_deviance
        """
        error_name = "Mean Tweedie Deviance"
        for method, combo in self.combos.items():

            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(met.mean_tweedie_deviance(true, pred))

    def mean_pinball_loss(self):
        """Calculate the mean pinball loss between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_pinball_loss
        """
        error_name = "Mean Pinball Deviance"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._errors[name]["Variable"]) == len(self._errors[name][method]):
                    self._errors[name]["Variable"].append(error_name)
                self._errors[name][method].append(met.mean_pinball_loss(true, pred))

    def return_errors(self):
        """Returns all calculated errors in dataframe format"""
        for key, item in self._errors.items():
            if not isinstance(self._errors[key], pd.DataFrame):
                self._errors[key] = pd.DataFrame(data=dict(item))
            if "Variable" in self._errors[key].columns:
                self._errors[key] = self._errors[key].set_index("Variable")
            self._errors[key] = self._errors[key].T
        self._errors = dict(self._errors)
        return self._errors

