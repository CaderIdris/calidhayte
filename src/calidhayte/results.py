from typing import Any

import pandas as pd
from sklearn import metrics as met
from sklearn.pipeline import Pipeline


class Results:
    """

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
        x: pd.DataFrame,
        y: pd.DataFrame,
        target: str,
        pipelines: dict[str, dict[str, dict[int, Pipeline]]],
    ):
        """
        """
        self.x = x
        self.y = y
        self.target = target
        self.pipelines = pipelines
        self._errors = pd.DataFrame()

    def _sklearn_error_meta(self, err: Any, name: str, **kwargs):
        """
        """
        idx = 0
        for technique, var_combos in self.pipelines.items():
            for vars, folds in var_combos.items():
                for fold, pipe in folds.items():
                    true = self.y.loc[
                            :, self.target
                            ][self.y.loc[:, 'Fold'] == fold]
                    pred_raw = self.x.loc[true.index, :]
                    pred = pipe.predict(pred_raw)
                    error = err(true, pred, **kwargs)
                    self._errors.loc[idx, 'Technique'] = technique
                    self._errors.loc[idx, 'Variables'] = vars
                    self._errors.loc[idx, 'Fold'] = fold
                    self._errors.loc[idx, name] = error
                    idx = idx+1

    def explained_variance_score(self):
        """Calculate the explained variance score between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.explained_variance_score
        """
        self._sklearn_error_meta(
                met.explained_variance_score,
                'Explained Variance Score'
                )

    def max(self):
        """Calculate the max error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.max_error
        """
        self._sklearn_error_meta(
                met.max_error,
                'Max Error'
                )

    def mean_absolute(self):
        """Calculate the mean absolute error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_absolute_error
        """
        self._sklearn_error_meta(
                met.mean_absolute_error,
                'Mean Absolute Error'
                )

    def root_mean_squared(self):
        """Calculate the root mean squared error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_squared_error
        """
        self._sklearn_error_meta(
                met.mean_squared_error,
                'Root Mean Squared Error',
                squared=False
                )

    def root_mean_squared_log(self):
        """Calculate the root mean squared log error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_squared_log_error
        """
        self._sklearn_error_meta(
                met.mean_squared_log_error,
                'Root Mean Squared Log Error',
                squared=False
                )

    def median_absolute(self):
        """Calculate the median absolute error between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.median_absolute_error
        """
        self._sklearn_error_meta(
                met.median_absolute_error,
                'Median Absolute Error'
                )

    def mean_absolute_percentage(self):
        """Calculate the mean absolute percentage error between the true
        values (y) and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_absolute_percentage_error
        """
        self._sklearn_error_meta(
                met.mean_absolute_percentage_error,
                'Mean Absolute Percentage Error'
                )

    def r2(self):
        """Calculate the r2 between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.r2_score
        """
        self._sklearn_error_meta(
                met.r2_score,
                'r2'
                )

    def mean_poisson_deviance(self):
        """Calculate the mean poisson deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_poisson_deviance
        """
        self._sklearn_error_meta(
                met.mean_poisson_deviance,
                'Mean Poisson Deviance'
                )

    def mean_gamma_deviance(self):
        """Calculate the mean gamma deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_gamma_deviance
        """
        self._sklearn_error_meta(
                met.mean_gamma_deviance,
                'Mean Gamma Deviance'
                )

    def mean_tweedie_deviance(self):
        """Calculate the mean tweedie deviance between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_tweedie_deviance
        """
        self._sklearn_error_meta(
                met.mean_tweedie_deviance,
                'Mean Tweedie Deviance'
                )

    def mean_pinball_loss(self):
        """Calculate the mean pinball loss between the true values (y)
        and predicted y (x)

        This technique is explained in further detail at:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.mean_pinball_loss
        """
        self._sklearn_error_meta(
                met.mean_pinball_loss,
                'Mean Pinball Deviance'
                )

    def return_errors(self) -> pd.DataFrame:
        """Returns all calculated errors in dataframe format"""
        return self._errors.set_index(['Technique', 'Variables', 'Fold'])
