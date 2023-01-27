import math
import re

import pandas as pd


class Calibrate:
    """
    Calibrates test and training data with the provided coefficients

    ```

    Attributes
    ----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    coefficients : pd.DataFrame
        Coefficients to calibrate with
    y_pred : dict[str, pd.DataFrame]
        Dictionary containing predicted y measurements derived from the test
        and train data after calibration with coefficients

    Methods
    -------
    _pymc_calibrate(coeffs)
        Calibrates signal using coefficients calculated with the pymc library

    _skl_calibrate(coeffs)
        Calibrates signal using coefficients calculated with the sklearn
        library

    return_measurements()
        Returns the calibrated measurements
    """
    def __init__(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            coefficients: pd.DataFrame
            ):
        """ Constructs the calibration object

        Parameters
        ----------
        train : pd.DataFrame
            Training dataset to calibrate
        test : pd.DataFrame
            Testing dataset to calibrate
        coefficients : pd.DataFrame
            Coefficients to calibrate datasets off of
        """
        empty_dataframe_test = list()
        for name, df in {
                "Train": train,
                "Test": test,
                "Coefficients": coefficients
                }.items():
            for shape_index in [0, 1]:
                if df.shape[shape_index] == 0:
                    empty_dataframe_test.append(f"{name} axis {shape_index}")

        if empty_dataframe_test:
            raise ValueError(
                f"The following axis are empty: {empty_dataframe_test}"
            )

        self.train = train
        self.test = test
        self.coefficients = coefficients
        self.y_pred: dict[str, pd.DataFrame] = dict()

        column_names = self.coefficients.columns
        pymc_calibration = any(
                [bool(re.search(r"^sd\.", col))
                 for col in column_names]
                )
        if pymc_calibration:
            self._pymc_calibrate()
        else:
            self._skl_calibrate()

    def _pymc_calibrate(self):
        """ Calibrates x measurements with provided pymc coefficients. Returns
        mean, max and min calibrations, where max and min are +-2*sd.

        Pymc calibrations don't just provide a coefficient for each variable
        in the form of a mean but also provide a standard deviation on that
        mean. By taking the mean coefficients, mean + 2*sd (max) and mean -
        2*sd (min) we get 3 potential values for the predicted y value.
        """
        for dset_name, dataset in {
                'Test': self.test,
                'Train': self.train
                }.items():
            for subset, std_mult in {
                    "Mean": 0,
                    "Minimum": -2,
                    "Maximum": 2
                    }.items():
                cal_dataset = pd.DataFrame()
                for coeff_set in self.coefficients.iterrows():
                    coeff_names = coeff_set[0]
                    coeff_vals = coeff_set[1]
                    for index, name in enumerate(coeff_names.split(' + ')):
                        if index == 0:
                            cal_dataset[coeff_names] = (
                                    dataset.loc[:, name] *
                                    coeff_vals.loc[f'coeff.{name}']
                                    ) + (
                                            coeff_vals.loc[f'coeff.{name}'] *
                                            coeff_vals.loc[f'sd.{name}'] *
                                            std_mult
                                        )

                        else:
                            cal_dataset[coeff_names] = (
                                    cal_dataset[coeff_names] +
                                    (
                                        dataset.loc[:, name] *
                                        coeff_vals.loc[f'coeff.{name}']
                                        )
                                    ) + (
                                            coeff_vals.loc[f'coeff.{name}'] *
                                            coeff_vals.loc[f'sd.{name}'] *
                                            std_mult
                                        )
                    cal_dataset[coeff_names] = (
                            cal_dataset[coeff_names] +
                            coeff_vals.loc['i.intercept']
                            ) + (
                                    coeff_vals.loc['i.intercept'] *
                                    coeff_vals.loc['sd.intercept'] *
                                    std_mult
                                )
                self.y_pred[f'{subset}.{dset_name}'] = cal_dataset

    def _skl_calibrate(self):
        """ Calibrate x measurements with provided skl coefficients. Returns
        skl calibration.

        Scikitlearn calibrations provide one coefficient for each variable,
        unlike pymc, so only one predicted signal is returned.

        """
        for dset_name, dataset in {
                'Test': self.test,
                'Train': self.train
                }.items():
            cal_dataset = pd.DataFrame()
            for coeff_set in self.coefficients.iterrows():
                coeff_names = coeff_set[0]
                coeff_vals = coeff_set[1]
                for index, name in enumerate(coeff_names.split(' + ')):
                    if index == 0:
                        cal_dataset[coeff_names] = (
                                dataset.loc[:, name] *
                                coeff_vals.loc[f'coeff.{name}']
                                )
                    else:
                        cal_dataset[coeff_names] = (
                                cal_dataset[coeff_names] +
                                (
                                    dataset.loc[:, name] *
                                    coeff_vals.loc[f'coeff.{name}']
                                    )
                                )
                cal_dataset[coeff_names] = (
                        cal_dataset[coeff_names] +
                        coeff_vals.loc['i.intercept']
                        )
            self.y_pred[dset_name] = cal_dataset

    def join_measurements(self) -> dict[str, pd.DataFrame]:
        """ Joins test and train measurements into one dataframe and sorts them
        by index to recreate the initial measurements

        Parameters
        ----------
        None

        Returns
        -------
        joined : dict[str, pd.DataFrame]
            A dictionary containing at least two keys.
            Uncalibrated - The uncalibrated measurements, each column represents
            measurement variable used in the calibration
            Calibrated - The calibrated measurements, each column represents a
            variable combination used to calibrate the measurements
            If PyMC was used to calibrate the measurements, Calibrated is
            prefixed by Minimum., Mean. and Maximum.

        """

        joined = dict()
        joined['Uncalibrated'] = pd.concat([self.test, self.train]).sort_index()

        pymc_bool = any(
                [
                    re.search(r"^Mean\.", key)
                    for key in self.y_pred.keys()
                    ]
                )

        if pymc_bool:
            joined['Mean.Calibrated'] = pd.concat(
                    [self.y_pred["Mean.Train"],
                     self.y_pred["Mean.Test"]]
                    ).sort_index()
            joined['Minimum.Calibrated'] = pd.concat(
                    [self.y_pred["Minimum.Train"],
                     self.y_pred["Minimum.Test"]]
                    ).sort_index()
            joined['Maximum.Calibrated'] = pd.concat(
                    [self.y_pred["Maximum.Train"],
                     self.y_pred["Maximum.Test"]]
                    ).sort_index()
        else:
            joined['Calibrated'] = pd.concat(
                    [self.y_pred["Train"],
                     self.y_pred["Test"]]
                    ).sort_index()

        return joined

    def return_measurements(self) -> dict[str, pd.DataFrame]:
        """ Returns the calibrated measurements

        Parameters
        ----------
        None

        Returns
        -------
        y_pred : dict[str, pd.DataFrame]
            All calibrated measurements, split into Train and Test data. If
            pymc was used to calibrate the data, Test and Train are also subset
            by "mean.", "min." and "max. prefixes respectively, representing
            the average, minimum and maximum signal calibrated with min and max
            calculated as 2 * std plus/minus the average calibration
            coefficient
        """
        return self.y_pred
