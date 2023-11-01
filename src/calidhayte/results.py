"""
Determine the performance of different calibration techniques using a range of
different metrics.

Acts as a wrapper for scikit-learn performance metrics [^skl].

[^skl]: https://scikit-learn.org/stable/modules/classes.html
"""

try:
    from typing import Any, TypeAlias
except ImportError:
    from typing_extensions import Any, TypeAlias

import pandas as pd
from sklearn import metrics as met
from sklearn.pipeline import Pipeline

CoefficientPipelineDict: TypeAlias = dict[str,  # Technique name
                                          dict[str,  # Scaling technique
                                               dict[str,  # Variable combo
                                                    dict[int,  # Fold
                                                         Pipeline]]]]
"""
Type alias for the nested dictionaries that the models are stored in
"""


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
        models: CoefficientPipelineDict
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
        self.errors: pd.DataFrame = pd.DataFrame()
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

    def _sklearn_error_meta(self, err: Any, name: str, **kwargs):
        """
        """
        idx = 0
        for technique, scaling_techniques in self.models.items():
            for scaling_technique, var_combos in scaling_techniques.items():
                for vars, folds in var_combos.items():
                    for fold, pipe in folds.items():
                        true = self.y.loc[
                                :, self.target
                                ][self.y.loc[:, 'Fold'] == fold]
                        pred_raw = self.x.loc[true.index, :]
                        pred = pipe.predict(pred_raw)
                        error = err(true, pred, **kwargs)
                        if idx not in self.errors.index:
                            self.errors.loc[idx, 'Technique'] = technique
                            self.errors.loc[
                                    idx, 'Scaling Method'
                                    ] = scaling_technique
                            self.errors.loc[idx, 'Variables'] = vars
                            self.errors.loc[idx, 'Fold'] = fold
                        self.errors.loc[idx, name] = error
                        idx = idx+1

    def explained_variance_score(self):
        """Calculate the explained variance score between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.explained_variance_score\
        )
        """
        self._sklearn_error_meta(
                met.explained_variance_score,
                'Explained Variance Score'
                )

    def max(self):
        """Calculate the max error between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.max_error\
        )
        """
        self._sklearn_error_meta(
                met.max_error,
                'Max Error'
                )

    def mean_absolute(self):
        """Calculate the mean absolute error between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.mean_absolute_error\
        )
        """
        self._sklearn_error_meta(
                met.mean_absolute_error,
                'Mean Absolute Error'
                )

    def root_mean_squared(self):
        """Calculate the root mean squared error between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.mean_squared_error\
        )
        """
        self._sklearn_error_meta(
                met.mean_squared_error,
                'Root Mean Squared Error',
                squared=False
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
                met.mean_squared_log_error,
                'Root Mean Squared Log Error',
                squared=False
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
                met.median_absolute_error,
                'Median Absolute Error'
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
                'Mean Absolute Percentage Error'
                )

    def r2(self):
        """Calculate the r2 between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.r2_score\
        )
        """
        self._sklearn_error_meta(
                met.r2_score,
                'r2'
                )

    def mean_poisson_deviance(self):
        """Calculate the mean poisson deviance between the true values (y)
        and predicted y (x) [^tech].

        [^tech]: [Link](\
        https://scikit-learn.org/stable/modules/generated/\
sklearn.metrics.mean_poisson_deviance\
        )
        """
        self._sklearn_error_meta(
                met.mean_poisson_deviance,
                'Mean Poisson Deviance'
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
                met.mean_gamma_deviance,
                'Mean Gamma Deviance'
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
                met.mean_tweedie_deviance,
                'Mean Tweedie Deviance'
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
                met.mean_pinball_loss,
                'Mean Pinball Deviance'
                )

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
        return self.errors.set_index(
                    ['Technique', 'Scaling Method', 'Variables', 'Fold']
                )
