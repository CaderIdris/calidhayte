import re
from typing import Literal, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics as met

from .calibrate import Calibrate

mpl.use("pgf")  # Used to make pgf files for latex
plt.rcParams.update({"figure.max_open_warning": 0})


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
        test_sets: list[
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
        self._errors: dict[str, pd.DataFrame] = dict()
        self._cal = Calibrate(
                self.train,
                self.test,
                self.coefficients
                )
        self.y_subsets = self._cal.return_measurements()
        self.y_full = self._cal.join_measurements()
        self._datasets = self._prepare_datasets(test_sets)
        self.x_name = x_name
        self.y_name = y_name

    def _prepare_datasets(
            self,
            test_sets: list[
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
                ]
            ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Prepare the datasets to be analysed

        Parameters
        ----------
        test_sets : list[str]
            List containing datasets to use
            Datasets are as follows:
                - Calibrated Train
                    Predicted y measurements from training set using
                    coefficients
                - Calibrated Test
                    Predicted y measurements from testing set using
                    coefficients
                - Calibrated Full
                    Predicted y measurements from both sets using
                    coefficients
                - Uncalibrated Train
                    Base x measurements from training set
                - Uncalibrated Test
                    Base x measurements from testing set
                - Uncalibrated Full
                    Base x measurements from testing set
                - Minimum
                    Mean bayesian coefficients - 2 times standard deviation
                - Maximum
                    Mean bayesian coefficients + 2 times standard deviation
        Returns
        -------
        dict[str, dict[str, pd.DataFrame]]
            Contains all subsets of the data to be analysed
        """
        uncalibrated_datasets = {
                "Uncalibrated Train": {
                    "x": self.train.loc[:, ['x']],
                    "y": self.train.loc[:, ['y']]
                    },
                "Uncalibrated Test": {
                    "x": self.test.loc[:, ['x']],
                    "y": self.test.loc[:, ['y']]
                    },
                "Uncalibrated Full": {
                    "x": self.y_full['Uncalibrated'].loc[:, ['x']],
                    "y": self.y_full['Uncalibrated'].loc[:, ['y']]
                    }
                }
        pymc_bool = all(
                [
                    any(["Mean." in key for key in self.y_subsets.keys()]),
                    any(["Min." in key for key in self.y_subsets.keys()]),
                    any(["Max." in key for key in self.y_subsets.keys()])
                    ]
                )
        sets_to_use: list[str] = list(
                filter(
                    lambda x: x not in ["Minimum", "Maximum"],
                    test_sets
                    )
                )
        if pymc_bool:
            all_datasets = uncalibrated_datasets | {
                    "Calibrated Train (Mean)": {
                        "x": self.y_subsets['Mean.Train'],
                        "y": self.train.loc[:, ['y']]
                        },
                    "Calibrated Test (Mean)": {
                        "x": self.y_subsets['Mean.Test'],
                        "y": self.test.loc[:, ['y']]
                        },
                    "Calibrated Full (Mean)": {
                        "x": self.y_full['Mean.Calibrated'],
                        "y": self.y_full['Uncalibrated'].loc[:, ['y']]
                        },
                    "Calibrated Train (Minimum)": {
                        "x": self.y_subsets['Minimum.Train'],
                        "y": self.train.loc[:, ['y']]
                        },
                    "Calibrated Test (Minimum)": {
                        "x": self.y_subsets['Minimum.Test'],
                        "y": self.test.loc[:, ['y']]
                        },
                    "Calibrated Full (Minimum)": {
                        "x": self.y_full['Minimum.Calibrated'],
                        "y": self.y_full['Uncalibrated'].loc[:, ['y']]
                        },
                    "Calibrated Train (Maximum)": {
                        "x": self.y_subsets['Maximum.Train'],
                        "y": self.train.loc[:, ['y']]
                        },
                    "Calibrated Test (Maximum)": {
                        "x": self.y_subsets['Maximum.Test'],
                        "y": self.test.loc[:, ['y']]
                        },
                    "Calibrated Full (Maximum)": {
                        "x": self.y_full['Maximum.Calibrated'],
                        "y": self.y_full['Uncalibrated'].loc[:, ['y']]
                        },
                    }
            min_max = filter(
                    lambda x: x in ["Minimum", "Maximum"],
                    test_sets
                    )
            pymc_sets_to_use = list(
                    filter(
                        lambda x: not bool(re.search(r"^Calibrated ", x)),
                        sets_to_use
                        )
                    )
            for sub_pymc in ["Mean"] + list(min_max):
                for cal_subset in filter(
                        lambda x: bool(re.search(r"^Calibrated ", x)),
                        sets_to_use
                        ):
                    pymc_sets_to_use.append(f"{cal_subset} ({sub_pymc})")
            sets_to_use = pymc_sets_to_use
        else:
            all_datasets = uncalibrated_datasets | {
                    "Calibrated Train": {
                        "x": self.y_subsets['Train'],
                        "y": self.train.loc[:, ['y']]
                        },
                    "Calibrated Test": {
                        "x": self.y_subsets['Test'],
                        "y": self.test.loc[:, ['y']]
                        },
                    "Calibrated Full": {
                        "x": self.y_full['Calibrated'],
                        "y": self.y_full['Uncalibrated'].loc[:, ['y']]
                        }
                    }

        return {
                key: item for key, item in all_datasets.items()
                if key in sets_to_use
                }

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

