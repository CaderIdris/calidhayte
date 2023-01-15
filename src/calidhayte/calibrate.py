import re
import typing

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
        self.train = train
        self.test = test
        self.coefficients = coefficients
        self.y_pred = dict()

        column_names = self.coefficients.columns
        pymc_calibration = any(
                [bool(re.search(r"^sd\.", col))
                 for col in column_names]
                )
        for coefficient_set in self.coefficients.iterrows():
            if pymc_calibration:
                self.y_pred[coefficient_set[0]] = self._pymc_calibrate(
                        coefficient_set[1]
                        )
            else:
                self.y_pred[coefficient_set[0]] = self._skl_calibrate(
                        coefficient_set[1]
                        )

    def _pymc_calibrate(
            self,
            coeffs: pd.Series
            ) -> typing.Dict[str, pd.DataFrame]:
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
            i.Intercept and sd.Intercept.

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
            not in ["coeff.x", "sd.x", "sd.Intercept", "i.Intercept", "index"]
        ]
        coefficient_keys = list()
        for key in coefficient_keys_raw:
            if re.match(r"coeff\.", key):
                coefficient_keys.append(re.sub(r"coeff\.", "", key))
        y_pred_train = self.train["x"] * coeffs.get("coeff.x")
        y_pred_test = self.test["x"] * coeffs.get("coeff.x")
        y_pred = {
            "mean.Train": pd.Series(y_pred_train),
            "min.Train": pd.Series(y_pred_train),
            "max.Train": pd.Series(y_pred_train),
            "mean.Test": pd.Series(y_pred_test),
            "min.Test": pd.Series(y_pred_test),
            "max.Test": pd.Series(y_pred_test),
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
        to_add_int = coeffs.get("i.Intercept")
        int_error = 2 * coeffs.get("sd.Intercept")

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
            ) -> typing.Dict[str, pd.DataFrame]:
        """ Calibrate x measurements with provided skl coefficients. Returns
        skl calibration.

        Scikitlearn calibrations provide one coefficient for each variable,
        unlike pymc, so only one predicted signal is returned.

        Parameters
        ----------
        coeffs : pd.Series
            All coefficients to be calibrated with, the coefficients are
            present with the coeff. prefix and the intercept is present under
            the i.Intercept tag

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
            if element not in ["coeff.x", "i.Intercept", "index"]
        ]
        coefficient_keys = list()
        for key in coefficient_keys_raw:
            if re.match(r"coeff\.", key):
                coefficient_keys.append(re.sub(r"coeff\.", "", key))
        y_pred = {
            "Train": pd.Series(self.train["x"]) * coeffs.get("coeff.x"),
            "Test": pd.Series(self.test["x"]) * coeffs.get("coeff.x"),
        }
        for coeff in coefficient_keys:
            to_add_test = self.test[coeff] * coeffs.get(f"coeff.{coeff}")
            to_add_train = self.train[coeff] * coeffs.get(f"coeff.{coeff}")
            y_pred["Test"] = y_pred["Test"] + to_add_test
            y_pred["Train"] = y_pred["Train"] + to_add_train
        to_add = coeffs.get("i.Intercept")
        y_pred["Test"] = y_pred["Test"] + to_add
        y_pred["Train"] = y_pred["Train"] + to_add
        return y_pred

    def return_measurements(self) -> typing.Dict[str, pd.DataFrame]:
        return self.y_pred
