"""
Determine the performance of different calibration techniques using a range of
different metrics.

Acts as a wrapper for scikit-learn performance metrics [^skl].

[^skl]: https://scikit-learn.org/stable/modules/classes.html
"""

import logging
from pathlib import Path
import pickle
from typing import Any, Optional, TypeAlias, Union

import numpy as np
import pandas as pd
from sklearn import metrics as met
from sklearn.pipeline import Pipeline

CoefficientPipelineDict: TypeAlias = dict[
    str,  # Technique name
    dict[
        str,  # Scaling technique
        dict[str, dict[int, Union[Path, Pipeline]]],  # Variable combo  # Fold
    ],
]
"""
Type alias for the nested dictionaries that the models are stored in
"""

logger = logging.getLogger(f"__main__.{__name__}")


def crmse(
    true: pd.Series, predicted: pd.Series, squared: bool = False, **kwargs
):
    """
    Calculate the centered root mean squared error between pred and true

    $\\sqrt{\\frac{1}{N}\\sum_{n=1}^{N}[(p_n-\\bar{p}) - (t_n-\\bar{t})]^2}$
    """
    centred_pred = predicted.sub(predicted.mean())
    centred_true = true.sub(true.mean())
    cmse = centred_pred.sub(centred_true).pow(2.0).mean()
    if squared:
        return cmse
    else:
        return np.sqrt(cmse)


def mean_bias_error(true: pd.Series, predicted: pd.Series, **kwargs):
    """
    Calculates the mean bias error between pred and true

    $\\frac{1}{N}\\sum_{n=1}^{N}[p_n - t_n]$
    """
    return predicted.sub(true).mean()


def ubrmse(true: pd.Series, predicted: pd.Series, **kwargs):
    """
    Calculates the unbiased root mean squared error between pred and true

    $\\sqrt{\\frac{1}{N}\\sum_{n=1}^{N}[(p_n - t_n)^2] - (\\frac{1}{N}\\sum_\
{n=1}^{N}[p_n - t_n])^2}$
    """
    mse = met.mean_squared_error(true, predicted)
    bias = mean_bias_error(true, predicted)
    return np.sqrt(mse - (bias**2))


def ref_mean(true: pd.Series, _: pd.Series, **kwargs):
    """
    Calculates mean of reference measurements, useful for
    normalising errors
    """
    return true.mean()


def ref_sd(true: pd.Series, _: pd.Series, **kwargs):
    """
    Calculates standard deviation of reference measurements, useful for
    normalising errors
    """
    return true.std()

def ref_mad(true: pd.Series, _: pd.Series, **kwargs):
    """
    Calculates standard deviation of reference measurements, useful for
    normalising errors
    """
    return true.sub(true.mean()).abs().mean()

def ref_range(true: pd.Series, _: pd.Series, **kwargs):
    """
    Calculates range of reference measurements, useful for
    normalising errors
    """
    return true.max() - true.min()


def ref_iqr(true: pd.Series, _: pd.Series, **kwargs):
    """
    Calculates interquartile range of reference measurements, useful for
    normalising errors
    """
    return true.quantile(0.75) - true.quantile(0.25)


class Results:
    """
    Determine performance of models using a range of metrics.

    Used to compare a range of different models that were fitted in the
    Calibrate class in `coefficients.py`.
    """

    def __init__(
        self,
        x_data: pd.DataFrame,
        y_data: pd.DataFrame,
        target: str,
        models: CoefficientPipelineDict,
        errors: Optional[pd.DataFrame] = None,
    ):
        """
        Initialises the class

        Parameters
        ----------
        x_data : pd.DataFrame
            Dependent measurements
        y_data : pd.DataFrame
            Independent measurements
        target : str
            Column name of the primary feature to use in calibration, must be
            the name of a column in both `x_data` and `y_data`.
        models : CoefficientPipelineDict
            The calibrated models.
        errors : pd.DataFrame
            Any previously calculated errors. Useful if you need to skip over
            previous calculations.
        """
        if target not in x_data.columns or target not in y_data.columns:
            not_in_x = target not in x_data.columns
            not_in_y = target not in y_data.columns
            raise ValueError(
                f"{target} not in {'x' if not_in_x else ''}"
                f"{' and ' if not_in_x and not_in_y else ''}"
                f"{'y' if not_in_y else ''}"
            )
        self.x: pd.DataFrame = x_data
        """Dependent measurements"""
        self.y: pd.DataFrame = y_data
        """Independent Measurements"""
        self.target: str = target
        """Column name of primary feature to use in calibration"""
        self.models: CoefficientPipelineDict = models
        """
        They are stored in a nested structure as
        follows:
        1. Primary Key, name of the technique (e.g Lasso Regression).
        2. Scaling technique (e.g Yeo-Johnson Transform).
        3. Combination of variables used or `target` if calibration is
        univariate (e.g "`target` + a + b).
        4. Fold, which fold was used excluded from the calibration. If data
        if 5-fold cross validated, a key of 4 indicates the data was
        trained on folds 0-3.

        ```mermaid
            stateDiagram-v2
              models --> Technique
              state Technique {
                [*] --> Scaling
                [*]: The calibration technique used
                [*]: (e.g "Lasso Regression")
                state Scaling {
                  [*] --> Variables
                  [*]: The scaling technique used
                  [*]: (e.g "Yeo-Johnson Transform")
                  state Variables {
                    [*] : The combination of variables used
                    [*] : (e.g "x + a + b")
                    [*] --> Fold
                    state Fold {
                     [*] : Which fold was excluded from training data
                     [*] : (e.g 4 indicates folds 0-3 were used to train)
                    }
                  }
                }
              }
        ```

        """
        if errors is not None:
            self.errors: pd.DataFrame = errors
        else:
            self.errors = pd.DataFrame()
        """
        Results of error metric valculations. Index increases sequentially
        by 1, columns contain the technique, scaling method, variables and
        fold for each row. It also contains a column for each metric.

        ||Technique|Scaling Method|Variables|Fold|Explained Variance Score|...\
        |Mean Absolute Percentage Error|
        |---|---|---|---|---|---|---|---|
        |0|Random Forest|Standard Scaling|x + a|0|0.95|...|0.05|
        |1|Theil-Sen|Yeo-JohnsonScaling|x + a + b|1|0.98|...|0.01|
        |...|...|...|...|...|...|...|...|
        |55|Extra Trees|None|x|2|0.43|...|0.52|
        """
        self.pred_vals: dict[str, dict[str, dict[str, pd.DataFrame]]] = dict()
        """
        """
        self.cached_error_length: int = self.errors.shape[0]

    def _sklearn_error_meta(self, err: Any, name: str, **kwargs):
        """ """
        idx = self.cached_error_length
        try:
            if (
                self.errors[
                    np.logical_and(
                        self.errors["Technique"] == "None",
                        self.errors["Fold"] == "Validation",
                    )
                ]
                .loc[:, name]
                .notna()
                .any(axis=None)
            ):
                pass
            else:
                raise KeyError
        except KeyError:
            true = self.y.loc[:, self.target][
                self.y.loc[:, "Fold"] == "Validation"
            ]
            pred_raw = self.x.loc[
                true.index,
                self.target
            ]
            predicted = pred_raw.dropna()
            error = err(true[predicted.index], predicted, **kwargs)
            if idx not in self.errors.index:
                self.errors.loc[idx, "Technique"] = "None"
                self.errors.loc[idx, "Scaling Method"] = "None"
                self.errors.loc[idx, "Variables"] = self.target
                self.errors.loc[idx, "Fold"] = "All"
                self.errors.loc[idx, "Count"] = true[predicted.index].shape[0]
            self.errors.loc[idx, name] = error
            idx = idx + 1
        true = self.y.loc[:, self.target][
            self.y.loc[:, "Fold"] == "Validation"
        ]
        for technique, scaling_techniques in self.models.items():
            if self.pred_vals.get(technique) is None:
                self.pred_vals[technique] = dict()
            for scaling_technique, var_combos in scaling_techniques.items():
                if self.pred_vals[technique].get(scaling_technique) is None:
                    self.pred_vals[technique][scaling_technique] = dict()
                for vars, folds in var_combos.items():
                    self.pred_vals[technique][scaling_technique][
                        vars
                    ] = pd.DataFrame(index=true.index)
                    try:
                        if (
                            self.errors.loc[
                                (self.errors["Technique"] == technique)
                                & (
                                    self.errors["Scaling Method"]
                                    == scaling_technique
                                )
                                & (self.errors["Variables"] == vars)
                            ]
                            .loc[:, name]
                            .notna()
                            .any(axis=None)
                        ):
                            continue
                    except KeyError:
                        pass
                    for fold, pipe in folds.items():
                        if (
                            fold
                            not in self.pred_vals[technique][
                                scaling_technique
                            ][vars].columns
                        ):
                            pred_raw = self.x.loc[
                            true.index, :
                            ]
                            if isinstance(pipe, Pipeline):
                                pipe_to_use = pipe
                            elif isinstance(pipe, Path):
                                with pipe.open("rb") as pkl:
                                    pipe_to_use = pickle.load(pkl)
                            else:
                                continue
                            pred_no_ind = pipe_to_use.predict(pred_raw)
                            self.pred_vals[technique][scaling_technique][vars][
                                fold
                            ] = pred_no_ind
                        try:
                            predicted = self.pred_vals[technique][
                                scaling_technique
                            ][vars][fold].dropna()
                            error = err(
                                true[predicted.index], predicted, **kwargs
                            )
                        except ValueError as exc:

                            logger.warning(
                                "%s, %s, %s, %s",
                                technique, scaling_technique, vars, fold
                            )
                            for arg in exc.args:
                                logger.warning(str(arg))
                            error = np.nan
                            predicted = pd.Series()

                        if idx not in self.errors.index:
                            self.errors.loc[idx, "Technique"] = technique
                            self.errors.loc[
                                idx, "Scaling Method"
                            ] = scaling_technique
                            self.errors.loc[idx, "Variables"] = vars
                            self.errors.loc[idx, "Fold"] = fold
                            self.errors.loc[idx, "Count"] = true[
                                predicted.index
                            ].shape[0]
                        self.errors.loc[idx, name] = error
                        idx = idx + 1
                    if idx not in self.errors.index:
                        self.errors.loc[idx, "Technique"] = technique
                        self.errors.loc[
                            idx, "Scaling Method"
                        ] = scaling_technique
                        self.errors.loc[idx, "Variables"] = vars
                        self.errors.loc[idx, "Fold"] = "All"
                        self.errors.loc[idx, "Count"] = true.shape[0]
                    predicted = (
                        self.pred_vals[technique][scaling_technique][vars]
                        .mean(axis=1)
                        .dropna()
                    )
                    error = err(
                        self.y.loc[predicted.index, self.target],
                        predicted,
                        **kwargs,
                    )
                    self.errors.loc[idx, name] = error
                    idx = idx + 1

    def explained_variance_score(self):
        """Calculate the explained variance score between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.explained_variance_score\
        )
        """
        self._sklearn_error_meta(
            met.explained_variance_score, "Explained Variance Score"
        )

    def max(self):
        """Calculate the max error between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.max_error\
        )
        """
        self._sklearn_error_meta(met.max_error, "Max Error")

    def mean_absolute(self):
        """Calculate the mean absolute error between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.mean_absolute_error\
        )
        """
        self._sklearn_error_meta(
            met.mean_absolute_error, "Mean Absolute Error"
        )

    def root_mean_squared(self):
        """Calculate the root mean squared error between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\any
sklearn.metrics.mean_squared_error\
        )
        """
        self._sklearn_error_meta(
            met.root_mean_squared_error, "Root Mean Squared Error"
        )

    def root_mean_squared_log(self):
        """Calculate the root mean squared log error between the true values
        (y) and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.mean_squared_log_error\
        )
        """
        self._sklearn_error_meta(
            met.root_mean_squared_log_error,
            "Root Mean Squared Log Error"
        )

    def median_absolute(self):
        """Calculate the median absolute error between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.median_absolute_error\
        )
        """
        self._sklearn_error_meta(
            met.median_absolute_error, "Median Absolute Error"
        )

    def mean_absolute_percentage(self):
        """Calculate the mean absolute percentage error between the true
        values (y) and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.mean_absolute_percentage_error\
        )
        """
        self._sklearn_error_meta(
            met.mean_absolute_percentage_error,
            "Mean Absolute Percentage Error",
        )

    def r2(self):
        """Calculate the r2 between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.r2_score\
        )
        """
        self._sklearn_error_meta(met.r2_score, "r2")

    def mean_poisson_deviance(self):
        """Calculate the mean poisson deviance between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.mean_poisson_deviance\
        )
        """
        self._sklearn_error_meta(
            met.mean_poisson_deviance, "Mean Poisson Deviance"
        )

    def mean_gamma_deviance(self):
        """Calculate the mean gamma deviance between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.mean_gamma_deviance\
        )
        """
        self._sklearn_error_meta(
            met.mean_gamma_deviance, "Mean Gamma Deviance"
        )

    def mean_tweedie_deviance(self):
        """Calculate the mean tweedie deviance between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.mean_tweedie_deviance\
        )
        """
        self._sklearn_error_meta(
            met.mean_tweedie_deviance, "Mean Tweedie Deviance"
        )

    def mean_pinball_loss(self):
        """Calculate the mean pinball loss between the true values (y)
        predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated\
/sklearn.metrics.mean_pinball_loss\
        )
        """
        self._sklearn_error_meta(
            met.mean_pinball_loss, "Mean Pinball Deviance"
        )

    def centered_rmse(self, **kwargs):
        """
        Calculate the centered root mean squared error between pred and true

        $\\sqrt{\\frac{1}{N}\\sum_{n=1}^{N}[(p_n-\\bar{p}) - \
(t_n-\\bar{t})]^2}$
        """
        self._sklearn_error_meta(
            crmse,
            "Centered Root Mean Squared Error",
            **kwargs
        )

    def unbiased_rmse(self):
        """
        Calculates the unbiased root mean squared error between pred and true

        $\\sqrt{\\frac{1}{N}\\sum_{n=1}^{N}[(p_n - t_n)^2] - (\\frac{1}{N}\\\
sum_{n=1}^{N}[p_n - t_n])^2}$
        """
        self._sklearn_error_meta(ubrmse, "Unbiased Root Mean Squared Error")

    def mbe(self):
        """
        Calculates the mean bias error between pred and true

        $\\frac{1}{N}\\sum_{n=1}^{N}[p_n - t_n]$
        """
        self._sklearn_error_meta(mean_bias_error, "Mean Bias Error")

    def ref_iqr(self):
        """
        Calculates interquartile range of reference measurements, useful for
        normalising errors
        """
        self._sklearn_error_meta(ref_iqr, "Reference Interquartile Range")

    def ref_mean(self):
        """
        Calculates mean of reference measurements, useful for
        normalising errors
        """
        self._sklearn_error_meta(ref_mean, "Reference Mean")

    def ref_range(self):
        """
        Calculates range of reference measurements, useful for
        normalising errors
        """
        self._sklearn_error_meta(ref_range, "Reference Range")

    def ref_sd(self):
        """
        Calculates standard deviation of reference measurements, useful for
        normalising errors
        """
        self._sklearn_error_meta(ref_sd, "Reference Standard Deviation")

    def ref_mad(self):
        """
        Calculates standard deviation of reference measurements, useful for
        normalising errors
        """
        self._sklearn_error_meta(ref_mad, "Reference Absolute Deviation")

    def return_errors(self) -> pd.DataFrame:
        """
        Returns all calculated errors in dataframe format

        Initially the error dataframe has the following structure:

        ||Technique|Scaling Method|Variables|Fold|Explained Variance Score|\
        ...|Mean Absolute Percentage Error|
        |---|---|---|---|---|---|---|---|
        |0|Random Forest|Standard Scaling|x + a|0|0.95|...|0.05|
        |1|Theil-Sen|Yeo-JohnsonScaling|x + a + b|1|0.98|...|0.01|
        |...|...|...|...|...|...|...|...|
        |55|Extra Trees|None|x|2|0.43|...|0.52|

        However, before returning the data, a new MultiIndex is built using
        the Technique, Scaling Method, Variables and Fold columns. This
        allows easy comparison of the different techniques by grouping on one
        or multiple levels of the MultiIndex.

        Returns
        -------
        pd.DataFrame
            Results dataframe in the following format:

            |||||Explained Variance Score|\
            ...|Mean Absolute Percentage Error|
            |---|---|---|---|---|---|---|
            |Random Forest|Standard Scaling|x + a|0|0.95|...|0.05|
            |Theil-Sen|Yeo-JohnsonScaling|x + a + b|1|0.98|...|0.01|
            |...|...|...|...|...|...|...|
            |Extra Trees|None|x|2|0.43|...|0.52|

        """
        return self.errors
