from collections import defaultdict
from pathlib import Path
import re
import sqlite3 as sql

import matplotlib as mpl

mpl.use("pgf")  # Used to make pgf files for latex
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.max_open_warning": 0})
import numpy as np
import pandas as pd
from sklearn import metrics as met


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
        train,
        test,
        coefficients,
        x_name=None,
        y_name=None,
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
        self._errors = defaultdict(lambda: defaultdict(list))
        self.y_pred = self._calibrate()
        self.combos = self._get_all_combos()
        self._plots = defaultdict(lambda: defaultdict(list))
        self.x_name = x_name
        self.y_name = y_name

    def _get_all_combos(self):
        """Return all possible combinations of datasets.

        This module has the capacity to test the errors of 6 datasets,
        comprised of (un)calibrated test/train/both measurements. There are
        also extra available for pymc calibrations as there is a mean, min and
        max predicted signal. This function generates all the combinations
        and puts them in a dict of lists. This allows for much easier looping
        over the different combinations when calculating errors.

        Keyword Arguments:
            None
        """
        combos = dict()
        for method, y_pred in self.y_pred.items():
            combos[method] = self._all_combos(y_pred)
        return combos

    def _all_combos(
        self, pred, to_use={"Calibrated Test": True, "Uncalibrated Test": True}
    ):
        """Addition to _get_all_combos to get cleaner code

        Keyword arguments:
            pred (dict): Dictionary containing all calibrated signals for a
            single variable combination (e.g x, RH, T)

            to_use (dict): Dict containing all different combos to be used. If
            key is present and corresponding value is True, combo is added to list.
            Keys can be:
                - Calibrated Test: The calibrated test data
                - Uncalibrated Test: The uncalibrated test data
                - Calibrated Train: The calibrated training data
                - Uncalibrated Train: The uncalibrated training data
                - Calibrated Full: The calibrated test + training data
                - Uncalibrated Full: The uncalibrated test + training data
                - MinMax: Use the minimum and maximum values generated by pymc
        """
        combos = list()
        if re.search("mean.", str(pred.keys())):
            if to_use.get("Calibrated Test", False):
                combos.append(
                    ("Calibrated Test Data (Mean)", pred["mean.Test"], self.test["y"])
                )
            if to_use.get("Calibrated Test", False) and to_use.get("MinMax", False):
                combos.append(
                    ("Calibrated Test Data (Min)", pred["min.Test"], self.test["y"])
                )
                combos.append(
                    ("Calibrated Test Data (Max)", pred["max.Test"], self.test["y"])
                )
            if to_use.get("Uncalibrated Test", False):
                combos.append(
                    ("Uncalibrated Test Data", self.test["x"], self.test["y"])
                )
            if to_use.get("Calibrated Train", False):
                combos.append(
                    (
                        "Calibrated Train Data (Mean)",
                        pred["mean.Train"],
                        self.train["y"],
                    )
                )
            if to_use.get("Calibrated Train", False) and to_use.get("MinMax", False):
                combos.append(
                    ("Calibrated Train Data (Min)", pred["min.Train"], self.train["y"])
                )
                combos.append(
                    ("Calibrated Train Data (Max)", pred["max.Train"], self.train["y"])
                )
            if to_use.get("Uncalibrated Train", False):
                combos.append(
                    ("Uncalibrated Train Data", self.train["x"], self.train["y"])
                )
            if to_use.get("Calibrated Full", False):
                combos.append(
                    (
                        "Calibrated Full Data (Mean)",
                        pd.concat([pred["mean.Train"], pred["mean.Test"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
            if to_use.get("Calibrated Full", False) and to_use.get("MinMax", False):
                combos.append(
                    (
                        "Calibrated Full Data (Min)",
                        pd.concat([pred["min.Train"], pred["min.Test"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
                combos.append(
                    (
                        "Calibrated Full Data (Max)",
                        pd.concat([pred["max.Train"], pred["max.Test"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
            if to_use.get("Uncalibrated Full", False):
                combos.append(
                    (
                        "Uncalibrated Full Data",
                        pd.concat([self.train["x"], self.test["x"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
        else:
            if to_use.get("Calibrated Test", False):
                combos.append(("Calibrated Test Data", pred["Test"], self.test["y"]))
            if to_use.get("Calibrated Train", False):
                combos.append(("Calibrated Train Data", pred["Train"], self.train["y"]))
            if to_use.get("Calibrated Full", False):
                combos.append(
                    (
                        "Calibrated Full Data",
                        pd.concat([pred["Train"], pred["Test"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
            if to_use.get("Uncalibrated Test", False):
                combos.append(
                    ("Uncalibrated Test Data", self.test["x"], self.test["y"])
                )
            if to_use.get("Uncalibrated Train", False):
                combos.append(
                    ("Uncalibrated Train Data", self.train["x"], self.train["y"])
                )
            if to_use.get("Uncalibrated Full", False):
                combos.append(
                    (
                        "Uncalibrated Full Data",
                        pd.concat([self.train["x"], self.test["x"]]),
                        pd.concat([self.train["y"], self.test["y"]]),
                    )
                )
        return combos

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

