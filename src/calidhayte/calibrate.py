import math
import re
from typing import Dict

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
        self.y_pred: Dict[str, pd.DataFrame] = dict()

        column_names = self.coefficients.columns
        pymc_calibration = any(
                [bool(re.search(r"^sd\.", col))
                 for col in column_names]
                )
        for coefficient_set in self.coefficients.iterrows():
            if pymc_calibration:
                coeffs = self._pymc_calibrate(
                        coefficient_set[1]
                        )
            else:
                coeffs = self._skl_calibrate(
                        coefficient_set[1]
                        )
            for key, measures in coeffs.items():
                df = self.y_pred.get(key, None)
                if df is None:
                    self.y_pred[key] = pd.DataFrame()
                    df = self.y_pred[key]
                df[coefficient_set[0]] = measures

    def _pymc_calibrate(
            self,
            coeffs: pd.Series
            ) -> Dict[str, pd.DataFrame]:
        """ Calibrates x measurements with provided pymc coefficients. Returns
        mean, max and min calibrations, where max and min are +-2*sd.

        Pymc calibrations don't just provide a coefficient for each variable
        in the form of a mean but also provide a standard deviation on that
        mean. By taking the mean coefficients, mean + 2*sd (max) and mean -
        2*sd (min) we get 3 potential values for the predicted y value.

        Parameters
        ----------
        coeffs : pd.Series
            All coefficients to be calibrated with, the
            mean.coeff and sd.coeff correspond to the coefficient mean and
            associated standard deviation. Intercept mean and sd is given with
            i.intercept and sd.intercept.

        Returns
        -------
        y_pred : dict[str, pd.DataFrame]
            dictionary containing calibrated Test and Train signals as
            pd.DataFrame
        """
        coefficient_keys_raw = list(coeffs.dropna().index)
        coefficient_keys_raw = [
            element
            for element in coefficient_keys_raw
            if element
            not in ["coeff.x", "sd.x", "sd.intercept", "i.intercept", "index"]
        ]
        coefficient_keys = list()
        for key in coefficient_keys_raw:
            if re.match(r"coeff\.", key):
                coefficient_keys.append(re.sub(r"coeff\.", "", key))
        if not math.isnan(coeffs.get("coeff.x")):
            y_pred_train = self.train.loc[:, "x"] * coeffs.get("coeff.x")
            y_pred_test = self.test.loc[:, "x"] * coeffs.get("coeff.x")
            init_error = 2 * coeffs.get("sd.x")
        else:
            y_pred_train = self.train.loc[:, "x"] * 0
            y_pred_test = self.test.loc[:, "x"] * 0
            init_error = 0
        y_pred = {
            "mean.Train": pd.Series(y_pred_train),
            "min.Train": pd.Series(y_pred_train) - init_error,
            "max.Train": pd.Series(y_pred_train) + init_error,
            "mean.Test": pd.Series(y_pred_test),
            "min.Test": pd.Series(y_pred_test) - init_error,
            "max.Test": pd.Series(y_pred_test) + init_error
        }
        for coeff in coefficient_keys:
            to_add_train = self.train[coeff] * coeffs.get(f"coeff.{coeff}")
            to_add_test = self.test[coeff] * coeffs.get(f"coeff.{coeff}")
            coeff_error_train = self.train[coeff] * \
                (2 * coeffs.get(f"sd.{coeff}"))
            coeff_error_test = self.test[coeff] * \
                (2 * coeffs.get(f"sd.{coeff}"))
            y_pred["mean.Train"] = y_pred["mean.Train"] + to_add_train
            y_pred["min.Train"] = y_pred["min.Train"] + (
                to_add_train - coeff_error_train
            )
            y_pred["max.Train"] = y_pred["max.Train"] + (
                to_add_train + coeff_error_train
            )
            y_pred["mean.Test"] = y_pred["mean.Test"] + to_add_test
            y_pred["min.Test"] = y_pred["min.Test"] + \
                (to_add_test - coeff_error_test)
            y_pred["max.Test"] = y_pred["max.Test"] + \
                (to_add_test + coeff_error_test)
        to_add_int = coeffs.get("i.intercept")
        int_error = 2 * coeffs.get("sd.intercept")

        y_pred["mean.Train"] = y_pred["mean.Train"] + to_add_int
        y_pred["min.Train"] = y_pred["min.Train"] + (to_add_int - int_error)
        y_pred["max.Train"] = y_pred["max.Train"] + (to_add_int + int_error)
        y_pred["mean.Test"] = y_pred["mean.Test"] + to_add_int
        y_pred["min.Test"] = y_pred["min.Test"] + (to_add_int - int_error)
        y_pred["max.Test"] = y_pred["max.Test"] + (to_add_int + int_error)
        return y_pred

    def _skl_calibrate(
            self,
            coeffs: pd.Series
            ) -> Dict[str, pd.DataFrame]:
        """ Calibrate x measurements with provided skl coefficients. Returns
        skl calibration.

        Scikitlearn calibrations provide one coefficient for each variable,
        unlike pymc, so only one predicted signal is returned.

        Parameters
        ----------
        coeffs : pd.Series
            All coefficients to be calibrated with, the coefficients are
            present with the coeff. prefix and the intercept is present under
            the i.intercept tag

        Returns
        -------
        y_pred : dict[str, pd.DataFrame]
            dictionary containing calibrated Test and Train signals as
            pd.DataFrame
        """
        coefficient_keys_raw = list(coeffs.dropna().index)
        coefficient_keys_raw = [
            element
            for element in coefficient_keys_raw
            if element not in ["coeff.x", "i.intercept", "index"]
        ]
        coefficient_keys = list()
        for key in coefficient_keys_raw:
            if re.match(r"coeff\.", key):
                coefficient_keys.append(re.sub(r"coeff\.", "", key))
        if not math.isnan(coeffs.get("coeff.x")):
            y_pred = {
                "Train": pd.Series(self.train["x"]) * coeffs.get("coeff.x"),
                "Test": pd.Series(self.test["x"]) * coeffs.get("coeff.x"),
            }
        else:
            y_pred = {
                "Train": pd.Series(self.train["x"]) * 0,
                "Test": pd.Series(self.test["x"]) * 0,
            }

        for coeff in coefficient_keys:
            to_add_test = self.test[coeff] * coeffs.get(f"coeff.{coeff}")
            to_add_train = self.train[coeff] * coeffs.get(f"coeff.{coeff}")
            y_pred["Test"] = y_pred["Test"] + to_add_test
            y_pred["Train"] = y_pred["Train"] + to_add_train
        to_add = coeffs.get("i.intercept")
        y_pred["Test"] = y_pred["Test"] + to_add
        y_pred["Train"] = y_pred["Train"] + to_add
        return y_pred

    def join_measurements(self) -> Dict[str, pd.DataFrame]:
        """ Joins test and train measurements into one dataframe and sorts them
        by index to recreate the initial measurements

        Parameters
        ----------
        None

        Returns
        -------
        joined : dict[str, pd.DataFrame]
            A dictionary containing at least two keys.
            x - The precalibrated measurements, each column represents an
            measurements variable used in the calibration
            y - The calibrated measurements, each column represents a
            variable combination used to calibrate the measurements
            Optional columns:
                y.min - Minimum calibrated value, if pymc used to calibrate
                y.max - Maximum calibrated value, if pymc used to calibrate

        """

        joined = dict()
        joined['x'] = pd.concat([self.test, self.train]).sort_index()

        print(self.y_pred)

        pymc_bool = any(
                [
                    re.search(r"mean\.", key)
                    for key in self.y_pred.keys()
                    ]
                )

        if pymc_bool:
            joined['y'] = pd.concat(
                    [self.y_pred["mean.Train"],
                     self.y_pred["mean.Test"]]
                    ).sort_index()
            joined['y.min'] = pd.concat(
                    [self.y_pred["min.Train"],
                     self.y_pred["min.Test"]]
                    ).sort_index()
            joined['y.max'] = pd.concat(
                    [self.y_pred["max.Train"],
                     self.y_pred["max.Test"]]
                    ).sort_index()
        else:
            joined['y'] = pd.concat(
                    [self.y_pred["Train"],
                     self.y_pred["Test"]]
                    ).sort_index()

        return joined

    def return_measurements(self) -> Dict[str, pd.DataFrame]:
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
