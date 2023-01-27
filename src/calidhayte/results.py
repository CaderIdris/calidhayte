from collections import defaultdict
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
        self._datasets = self._prepare_datasets(datasets_to_use)
        self.x_name = x_name
        self.y_name = y_name

    def _prepare_datasets(
            self,
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
                ]
            ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Prepare the datasets to be analysed

        Parameters
        ----------
        datasets_to_use : list[str]
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
                    any(["Minimum." in key for key in self.y_subsets.keys()]),
                    any(["Maximum." in key for key in self.y_subsets.keys()])
                    ]
                )
        if pymc_bool:
            cal_datasets = {
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
        else:
            cal_datasets = {
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
        datasets = uncalibrated_datasets | cal_datasets
        uncal_sets = filter(
                lambda x: bool(re.search(r'^Uncalibrated ', x)),
                datasets_to_use
                )
        cal_sets = filter(
                lambda x: bool(re.search(r'^Calibrated ', x)),
                datasets_to_use
                )
        selected_datasets = dict()
        for uncal_key in uncal_sets:
            selected_datasets[str(uncal_key)] = datasets[uncal_key]
        if pymc_bool:
            min_max_sets = filter(
                    lambda x: bool(re.search(r'^Minimum|^Maximum', x)),
                    datasets_to_use
                    )
            for cal_key in cal_sets:
                for pymc_subset in ['Mean'] + list(min_max_sets):
                    selected_datasets[
                            f'{cal_key} ({pymc_subset})'
                            ] = datasets[f'{cal_key} ({pymc_subset})']
        else:
            for cal_key in cal_sets:
                selected_datasets[str(cal_key)] = datasets[cal_key]

        return selected_datasets

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
                        "Explained Variance Score", var
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
                        "Max Error", var
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
                        "Mean Absolute Error", var
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
                        "Root Mean Squared Error", var
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
                        "Root Mean Squared Log Error", var
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
                        "Median Absolute Error", var
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
                        "Mean Absolute Percentage Error", var
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
                        "r2", var
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
                        "Mean Poisson Deviance", var
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
                        "Mean Gamma Deviance", var
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
                        "Mean Tweedie Deviance", var
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
                        "Mean Pinball Deviance", var
                        ] = met.mean_pinball_loss(
                                dset['y'].loc[:, 'y'],
                                dset['x'].loc[:, var]
                                )

    def return_errors(self) -> dict[str, pd.DataFrame]:
        """Returns all calculated errors in dataframe format"""
        return dict(self._errors)
