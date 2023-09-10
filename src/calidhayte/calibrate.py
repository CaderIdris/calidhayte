""" Contains code used to perform a range of univariate and multivariate
regressions on provided data.

Acts as a wrapper for scikit-learn [^skl], XGBoost [^xgb] and PyMC (via Bambi)
[^pmc]

[^skl]: https://scikit-learn.org/stable/modules/classes.html
[^xgb]: https://xgboost.readthedocs.io/en/stable/python/python_api.html
[^pmc]: https://bambinos.github.io/bambi/api/
"""

from collections.abc import Iterable
from copy import deepcopy as dc
import logging
import sys
from typing import Any, Literal, Union
import warnings

# import bambi as bmb
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import cross_decomposition as cd
from sklearn import ensemble as en
from sklearn import gaussian_process as gp
from sklearn import isotonic as iso
from sklearn import linear_model as lm
from sklearn import neural_network as nn
from sklearn import svm
from sklearn import tree
import sklearn.preprocessing as pre
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

_logger = logging.getLogger("pymc")
_logger.setLevel(logging.ERROR)


def cont_strat_folds(
        df: pd.DataFrame,
        target_var: str,
        splits: int = 5,
        strat_groups: int = 5,
        seed: int = 62
        ) -> pd.DataFrame:
    """
    Creates stratified k-folds on continuous variable
    ----------
    df : pd.DataFrame
        Target data to stratify on.
    target_var : str
        Target feature name.
    splits : int, default=5
        Number of folds to make.
    strat_groups : int, default=10
        Number of groups to split data in to for stratification.
    seed : int, default=62
        Random state to use.

    Returns
    -------
    pd.DataFrame
        `y_df` with added 'Fold' column, specifying which test data fold
        variable corresponds to.

    Examples
    --------
    >>> df = pd.read_csv('data.csv')
    >>> df
    |    | x | a | b |
    |    |   |   |   |
    |  0 |2.3|1.8|7.2|
    |  1 |3.2|9.6|4.5|
    |....|...|...|...|
    |1000|2.3|4.5|2.2|
    >>> df_with_folds = const_strat_folds(
            df=df,
            target='a',
            splits=3,
            strat_groups=3.
            seed=78
        )
    >>> df_with_folds
    |    | x | a | b |Fold|
    |    |   |   |   |    |
    |  0 |2.3|1.8|7.2| 2  |
    |  1 |3.2|9.6|4.5| 1  |
    |....|...|...|...|....|
    |1000|2.3|4.5|2.2| 0  |

    All folds should have a roughly equal distribution of values for 'a'

    """
    _df = df.copy()
    _df['Fold'] = -1
    skf = StratifiedKFold(
            n_splits=splits,
            random_state=seed,
            shuffle=True
            )
    _df['Group'] = pd.cut(
            _df.loc[:, target_var],
            strat_groups,
            labels=False
            )
    group_label = _df.loc[:, 'Group']

    for fold_number, (_, v) in enumerate(skf.split(group_label, group_label)):
        _df.loc[v, 'Fold'] = fold_number
    return _df.drop('Group', axis=1)


class Calibrate:
    """
    Calibrate x against y using a range of different methods provided by
    scikit-learn[^skl], xgboost[^xgb] and PyMC (via Bambi)[^pmc].

    [^skl]: https://scikit-learn.org/stable/modules/classes.html
    [^xgb]: https://xgboost.readthedocs.io/en/stable/python/python_api.html
    [^pmc]: https://bambinos.github.io/bambi/api/
    """

    def __init__(
            self,
            x_data: pd.DataFrame,
            y_data: pd.DataFrame,
            target: str,
            folds: int = 5,
            strat_groups: int = 10,
            scaler: Union[
                Iterable[
                    Literal[
                        'None',
                        'Standard Scale',
                        'MinMax Scale',
                        'Yeo-Johnson Transform'
                        'Box-Cox Transform',
                        'Quantile Transform (Uniform)',
                        'Quantile Transform (Gaussian)'
                        ]
                    ],
                Literal[
                    'All',
                    'None',
                    'Standard Scale',
                    'MinMax Scale',
                    'Yeo-Johnson Transform'
                    'Box-Cox Transform',
                    'Quantile Transform (Uniform)',
                    'Quantile Transform (Gaussian)',
                    ]
                ] = 'None',
            seed: int = 62
                 ):
        """Initialises class

        Used to compare one set of measurements against another.
        It can perform both univariate and multivariate regression, though
        some techniques can only do one or the other. Multivariate regression
        can only be performed when secondary variables are provided.

        Parameters
        ----------
        x_data : pd.DataFrame
            Data to be calibrated.
        y_data : pd.DataFrame
            'True' data to calibrate against.
        target : str
            Column name of the primary feature to use in calibration, must be
            the name of a column in both `x_data` and `y_data`.
        folds : int, default=5
            Number of folds to split the data into, using stratified k-fold.
        strat_groups : int, default=10
            Number of groups to stratify against, the data will be split into
            n equally sized bins where n is the value of `strat_groups`.
        scaler : iterable of {<br>\
            'None',<br>\
            'Standard Scale',<br>\
            'MinMax Scale',<br>\
            'Yeo-Johnson Transform',<br>\
            'Box-Cox Transform',<br>\
            'Quantile Transform (Uniform)',<br>\
            'Quantile Transform (Gaussian)',<br>\
            } or {<br>\
            'All',<br>\
            'None',<br>\
            'Standard Scale',<br>\
            'MinMax Scale',<br>\
            'Yeo-Johnson Transform',<br>\
            'Box-Cox Transform',<br>\
            'Quantile Transform (Uniform)',<br>\
            'Quantile Transform (Gaussian)',<br>\
            }, default='None'
            The scaling/transform method (or list of methods) to apply to the
            data
        seed : int, default=62
            Random state to use when shuffling and splitting the data into n
            folds. Ensures repeatability.

        Raises
        ------
        ValueError
            Raised if the target variables (e.g. 'NO2') is not a column name in
            both dataframes.
            Raised if `scaler` is not str, tuple or list

        Examples
        --------
        >>> from calidhayte.calibrate import Calibrate
        >>> import pandas as pd
        >>>
        >>> x = pd.read_csv('independent.csv')
        >>> x
        |   | a | b |
        | 0 |2.3|3.2|
        | 1 |3.4|3.1|
        |...|...|...|
        |100|3.7|2.1|
        >>>
        >>> y = pd.read_csv('dependent.csv')
        >>> y
        |   | a |
        | 0 |7.8|
        | 1 |9.9|
        |...|...|
        |100|9.5|
        >>>
        >>> calibration = Calibrate(
            x_data=x,
            y_data=y,
            target='a',
            folds=5,
            strat_groups=5,
            scaler = [
                'Standard Scale',
                'MinMax Scale'
                ],
            seed=62
        )
        >>> calibration.linreg()
        >>> calibration.lars()
        >>> calibration.omp()
        >>> calibration.ransac()
        >>> calibration.random_forest()
        >>>
        >>> models = calibration.return_models()
        >>> list(models.keys())
        [
            'Linear Regression',
            'Least Angle Regression',
            'Orthogonal Matching Pursuit',
            'RANSAC',
            'Random Forest'
        ]
        >>> list(models['Linear Regression'].keys())
        ['Standard Scale', 'MinMax Scale']
        >>> list(models['Linear Regression']['Standard Scale'].keys())
        ['a', 'a + b']
        >>> list(models['Linear Regression']['Standard Scale']['a'].keys())
        [0, 1, 2, 3, 4]
        >>> type(models['Linear Regression']['Standard Scale']['a'][0])
        <class sklearn.pipeline.Pipeline>
        >>> pipeline = models['Linear Regression']['Standard Scale']['a'][0]
        >>> x_new = pd.read_csv('independent_new.csv')
        >>> x_new
        |   | a | b |
        | 0 |3.5|2.7|
        | 1 |4.0|1.1|
        |...|...|...|
        |100|2.3|2.1|
        >>> pipeline.transform(x_new)
        |   | a |
        | 0 |9.7|
        | 1 |9.1|
        |...|...|
        |100|6.7|


        """
        if target not in x_data.columns or target not in y_data.columns:
            raise ValueError(
                    f"{target} does not exist in both columns."
                             )
        join_index = x_data.join(
                y_data,
                how='inner',
                lsuffix='x',
                rsuffix='y'
                ).dropna().index
        """
        The common indices between `x_data` and `y_data`, excluding missing
        values
        """
        self.x_data: pd.DataFrame = x_data.loc[join_index, :]
        """
        The data to be calibrated.
        """
        self.target: str = target
        """
        The name of the column in both `x_data` and `y_data` that
        will be used as the x and y variables in the calibration.
        """
        self.scaler_list: dict[str, Any] = {
                'None': None,
                'Standard Scale': pre.StandardScaler(),
                'MinMax Scale': pre.MinMaxScaler(),
                'Yeo-Johnson Transform': pre.PowerTransformer(
                    method='yeo-johnson'
                    ),
                'Box-Cox Transform': pre.PowerTransformer(method='box-cox'),
                'Quantile Transform (Uniform)': pre.QuantileTransformer(
                    output_distribution='uniform'
                    ),
                'Quantile Transform (Gaussian)': pre.QuantileTransformer(
                    output_distribution='normal'
                    )
                }
        """
        Keys for scaling algorithms available in the pipelines
        """
        self.scaler: list[str] = list()
        """
        The scaling algorithm(s) to preprocess the data with
        """
        if isinstance(scaler, str):
            if scaler == "All":
                if not bool(self.x_data.ge(0).all(axis=None)):
                    warnings.warn(
                        f'Box-Cox is not compatible with provided measurements'
                    )
                    self.scaler_list.pop('Box-Cox Transform')
                self.scaler.extend(self.scaler_list.keys())
            elif scaler in self.scaler_list.keys():
                self.scaler.append(scaler)
            else:
                self.scaler.append('None')
                warnings.warn(f'Scaling algorithm {scaler} not recognised')
        elif isinstance(scaler, (tuple, list)):
            for sc in scaler:
                if sc == 'Box-Cox Transform' and not bool(
                    self.x_data.ge(0).all(axis=None)
                ):
                    warnings.warn(
                        f'Box-Cox is not compatible with provided measurements'
                    )
                    continue
                if sc in self.scaler_list.keys():
                    self.scaler.append(sc)
                else:
                    warnings.warn(f'Scaling algorithm {sc} not recognised')
        else:
            raise ValueError('scaler parameter should be string, list or tuple')
        if not self.scaler:
            warnings.warn(
                f'No valid scaling algorithms provided, defaulting to None'
            )
            self.scaler.append('None')

        self.y_data = cont_strat_folds(
                y_data.loc[join_index, :],
                target,
                folds,
                strat_groups,
                seed
                )
        """
        The data that `x_data` will be calibrated against. A '*Fold*'
        column is added using the `const_strat_folds` function which splits
        the data into k stratified folds (where k is the value of
        `folds`). It splits the continuous measurements into n bins (where n
        is the value of `strat_groups`) and distributes each bin equally
        across all folds. This significantly reduces the chances of one fold
        containing a skewed distribution relative to the whole dataset.
        """
        self.models: dict[str,  # Technique name
                          dict[str,  # Scaling technique
                               dict[str,  # Variable combo
                                    dict[int,  # Fold
                                         Pipeline]]]] = dict()
        """
        The calibrated models. They are stored in a nested structure as
        follows:
        1. Primary Key, name of the technique (e.g Lasso Regression).
        2. Scaling technique (e.g Yeo-Johnson Transform).
        3. Combination of variables used or `target` if calibration is
        univariate (e.g "`target` + a + b).
        4. Fold, which fold was used excluded from the calibration. If data
        if 5-fold cross validated, a key of 4 indicates the data was trained on
        folds 0-3.

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

    def _sklearn_regression_meta(
            self,
            reg: Union[skl.base.RegressorMixin, Literal['t', 'gaussian']],
            name: str,
            min_coeffs: int = 1,
            max_coeffs: int = (sys.maxsize * 2) + 1,
            **kwargs
            ):
        """
        Metaclass, formats data and uses sklearn classifier to
        fit x to y

        Parameters
        ----------
        reg : sklearn.base.RegressorMixin or str
            Classifier to use, or distribution family to use for bayesian.
        name : str
            Name of classification technique to save pipeline to.
        min_coeffs : int, default=1
            Minimum number of coefficients for technique.
        max_coeffs : int, default=(sys.maxsize * 2) + 1
            Maximum number of coefficients for technique.

        Raises
        ------
        NotImplementedError
            PyMC currently doesn't work, TODO
        """
        x_secondary_cols = self.x_data.drop(self.target, axis=1).columns
        # All columns in x_data that aren't the target variable
        products = [[np.nan, col] for col in x_secondary_cols]
        secondary_vals = pd.MultiIndex.from_product(products)
        # Get all possible combinations of secondary variables in a pandas
        # MultiIndex
        if self.models.get(name) is None:
            self.models[name] = dict()
            # If the classification technique hasn't been used yet,
            # add its key to the models dictionary
        for scaler in self.scaler:
            if self.models[name].get(scaler) is None:
                self.models[name][scaler] = dict()
                # If the scaling technique hasn't been used with the classification
                # technique yet, add its key to the nested dictionary
            for sec_vals in secondary_vals:
                # Loop over all combinations of secondary values
                vals = [self.target] + [v for v in sec_vals if v == v]
                vals_str = ' + '.join(vals)
                if len(vals) < min_coeffs or len(vals) > max_coeffs:
                    # Skip if number of coeffs doesn't lie within acceptable range
                    # for technique. For example, isotonic regression
                    # only works with one variable
                    continue
                self.models[name][scaler][vals_str] = dict()
                for fold in self.y_data.loc[:, 'Fold'].unique():
                    y_data = self.y_data[
                            self.y_data.loc[:, 'Fold'] != fold
                            ]
                    if reg in ['t', 'gaussian']:
                        # If using PyMC bayesian model,
                        # format data and build model using bambi
                        # then store result in pipeline
                        # Currently doesn't work as PyMC models
                        # can't be pickled, so don't function with deepcopy. Needs
                        # looking into
                        raise NotImplementedError(
                            "PyMC functions currently don't work with deepcopy"
                        )
    #                    sc = scalers[scaler]
    #                    if sc is not None:
    #                        x_data = sc.fit_transform(
    #                                self.x_data.loc[y_data.index, :]
    #                                )
    #                    else:
    #                        x_data = self.x_data.loc[y_data.index, :]
    #                    x_data['y'] = y_data.loc[:, self.target]
    #                    model = bmb.Model(
    #                            f"y ~ {vals_str}",
    #                            x_data,
    #                            family=reg
    #                            )
    #                    _ = model.fit(
    #                        progressbar=False,
    #                        **kwargs
    #                        )
    #                    pipeline = Pipeline([
    #                        ("Scaler", scaler),
    #                        ("Regression", model)
    #                        ])
                    else:
                        # If using scikit-learn API compatible classifier,
                        # Build pipeline and fit to
                        pipeline = Pipeline([
                            ("Selector", ColumnTransformer([
                                    ("selector", "passthrough", vals)
                                ], remainder="drop")
                             ),
                            ("Scaler", self.scaler_list[scaler]),
                            ("Regression", reg)
                            ])
                        pipeline.fit(
                                self.x_data.loc[y_data.index, :],
                                y_data.loc[:, self.target]
                                )
                    self.models[name][scaler][vals_str][fold] = dc(pipeline)

    def pymc_bayesian(
            self,
            family: Literal[
                "Gaussian",
                "Student T",
            ] = "Gaussian",
            name: str = " PyMC Bayesian",
            **kwargs
            ):
        """
        Performs bayesian linear regression (either uni or multivariate)
        fitting x on y.

        Performs bayesian linear regression, both univariate and multivariate,
        on X against y. More details can be found at:
        https://pymc.io/projects/examples/en/latest/generalized_linear_models/
        GLM-robust.html

        Parameters
        ----------
        family : {'Gaussian', 'Student T'}, default='Gaussian'
            Statistical distribution to fit measurements to. Options are:
                - Gaussian
                - Student T
        """
        # Define model families
        model_families = {
            "Gaussian": "gaussian",
            "Student T": "t"
        }
        self._sklearn_regression_meta(
                model_families[family],
                f'{name} ({model_families})',
                **kwargs
        )

    def linreg(self, name: str = "Linear Regression", **kwargs):
        """
        Fit x on y via linear regression

        Parameters
        ----------
        name : str, default="Linear Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                lm.LinearRegression(**kwargs),
                name
                )

    def ridge(self, name: str = "Ridge Regression", **kwargs):
        """
        Fit x on y via ridge regression

        Parameters
        ----------
        name : str, default="Ridge Regression"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.Ridge(**kwargs),
                name
                )

    def ridge_cv(
            self,
            name: str = "Ridge Regression (Cross Validated)",
            **kwargs
            ):
        """
        Fit x on y via cross-validated ridge regression

        Parameters
        ----------
        name : str, default="Ridge Regression (Cross Validated)"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.RidgeCV(**kwargs),
                name
                )

    def lasso(self, name: str = "Lasso Regression", **kwargs):
        """
        Fit x on y via lasso regression

        Parameters
        ----------
        name : str, default="Lasso Regression"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.Lasso(**kwargs),
                name
                )

    def lasso_cv(
            self,
            name: str = "Lasso Regression (Cross Validated)",
            **kwargs
            ):
        """
        Fit x on y via cross-validated lasso regression

        Parameters
        ----------
        name : str, default="Lasso Regression (Cross Validated)"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.LassoCV(**kwargs),
                name
                )

    def multi_task_lasso(
            self,
            name: str = "Multi-task Lasso Regression",
            **kwargs
            ):
        """
        Fit x on y via multitask lasso regression

        Parameters
        ----------
        name : str, default="Multi-task Lasso Regression"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.MultiTaskLasso(**kwargs),
                name
                )

    def multi_task_lasso_cv(
            self,
            name: str = "Multi-task Lasso Regression (Cross Validated)",
            **kwargs
            ):
        """
        Fit x on y via cross validated multitask lasso regression

        Parameters
        ----------
        name : str, default="Multi-task Lasso Regression (Cross Validated)"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.MultiTaskLassoCV(**kwargs),
                name
                )

    def elastic_net(self, name: str = "Elastic Net Regression", **kwargs):
        """
        Fit x on y via elastic net regression

        Parameters
        ----------
        name : str, default="Elastic Net Regression"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.ElasticNet(**kwargs),
                name
                )

    def elastic_net_cv(
            self,
            name: str = "Elastic Net Regression (Cross Validated)",
            **kwargs
            ):
        """
        Fit x on y via cross validated elastic net regression

        Parameters
        ----------
        name : str, default="Elastic Net Regression (Cross Validated)"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.ElasticNetCV(**kwargs),
                name
                )

    def multi_task_elastic_net(
            self,
            name: str = "Multi-Task Elastic Net Regression",
            **kwargs
            ):
        """
        Fit x on y via multi-task elastic net regression

        Parameters
        ----------
        name : str, default="Multi-task Elastic Net Regression"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.MultiTaskElasticNet(**kwargs),
                name
                )

    def multi_task_elastic_net_cv(
            self,
            name: str = "Multi-Task Elastic Net Regression (Cross Validated)",
            **kwargs
            ):
        """
        Fit x on y via cross validated multi-task elastic net regression

        Parameters
        ----------
        name : str, default="Multi-Task Elastic Net Regression\
        (Cross Validated)"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.MultiTaskElasticNetCV(**kwargs),
                name
                )

    def lars(self, name: str = "Least Angle Regression", **kwargs):
        """
        Fit x on y via least angle regression

        Parameters
        ----------
        name : str, default="Least Angle Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                lm.Lars(**kwargs),
                name
                )

    def lars_lasso(
            self,
            name: str = "Least Angle Regression (Lasso)",
            **kwargs
            ):
        """
        Fit x on y via lasso least angle regression

        Parameters
        ----------
        name : str, default="Least Angle Regression (Lasso)"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.LassoLars(**kwargs),
                name
                )

    def omp(self, name: str = "Orthogonal Matching Pursuit", **kwargs):
        """
        Fit x on y via orthogonal matching pursuit regression

        Parameters
        ----------
        name : str, default="Orthogonal Matching Pursuit"
            Name of classification technique
        """
        self._sklearn_regression_meta(
                lm.OrthogonalMatchingPursuit(**kwargs),
                name,
                min_coeffs=2
                )

    def bayesian_ridge(
                self,
                name: str = "Bayesian Ridge Regression",
                **kwargs
            ):
        """
        Fit x on y via bayesian ridge regression

        Parameters
        ----------
        name : str, default="Bayesian Ridge Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                lm.BayesianRidge(**kwargs),
                name
                )

    def bayesian_ard(
            self,
            name: str = "Bayesian Automatic Relevance Detection",
            **kwargs
            ):
        """
        Fit x on y via bayesian automatic relevance detection

        Parameters
        ----------
        name : str, default="Bayesian Automatic Relevance Detection"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                lm.ARDRegression(**kwargs),
                name
                )

    def tweedie(self, name: str = "Tweedie Regression", **kwargs):
        """
        Fit x on y via tweedie regression

        Parameters
        ----------
        name : str, default="Tweedie Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                lm.TweedieRegressor(**kwargs),
                name
                )

    def stochastic_gradient_descent(
            self,
            name: str = "Stochastic Gradient Descent",
            **kwargs
            ):
        """
        Fit x on y via stochastic gradient descent regression

        Parameters
        ----------
        name : str, default="Stochastic Gradient Descent"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                lm.SGDRegressor(**kwargs),
                name
                )

    def passive_aggressive(
            self,
            name: str = "Passive Agressive Regression",
            **kwargs
            ):
        """
        Fit x on y via passive aggressive regression

        Parameters
        ----------
        name : str, default="Passive Agressive Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                lm.PassiveAggressiveRegressor(**kwargs),
                name
                )

    def ransac(self, name: str = "RANSAC", **kwargs):
        """
        Fit x on y via RANSAC regression

        Parameters
        ----------
        name : str, default="RANSAC"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                lm.RANSACRegressor(**kwargs),
                name
                )

    def theil_sen(self, name: str = "Theil-Sen Regression", **kwargs):
        """
        Fit x on y via theil-sen regression

        Parameters
        ----------
        name : str, default="Theil-Sen Regression"
            Name of classification technique.
        -Sen Regression
        """
        self._sklearn_regression_meta(
                lm.TheilSenRegressor(**kwargs),
                name
                )

    def huber(self, name: str = "Huber Regression", **kwargs):
        """
        Fit x on y via huber regression

        Parameters
        ----------
        name : str, default="Huber Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                lm.HuberRegressor(**kwargs),
                name
                )

    def quantile(self, name: str = "Quantile Regression", **kwargs):
        """
        Fit x on y via quantile regression

        Parameters
        ----------
        name : str, default="Quantile Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                lm.QuantileRegressor(**kwargs),
                name
                )

    def decision_tree(self, name: str = "Decision Tree", **kwargs):
        """
        Fit x on y using a decision tree

        Parameters
        ----------
        name : str, default="Decision Tree"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                tree.DecisionTreeRegressor(**kwargs),
                name
                )

    def extra_tree(self, name: str = "Extra Tree", **kwargs):
        """
        Fit x on y using an extra tree

        Parameters
        ----------
        name : str, default="Extra Tree"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                tree.ExtraTreeRegressor(**kwargs),
                name
                )

    def random_forest(self, name: str = "Random Forest", **kwargs):
        """
        Fit x on y using a random forest

        Parameters
        ----------
        name : str, default="Random Forest"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                en.RandomForestRegressor(**kwargs),
                name
                )

    def extra_trees_ensemble(
            self,
            name: str = "Extra Trees Ensemble",
            **kwargs
            ):
        """
        Fit x on y using an ensemble of extra trees

        Parameters
        ----------
        name : str, default="Extra Trees Ensemble"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                en.ExtraTreesRegressor(**kwargs),
                name
                )

    def gradient_boost_regressor(
            self,
            name: str = "Gradient Boosting Regression",
            **kwargs
            ):
        """
        Fit x on y using gradient boosting regression

        Parameters
        ----------
        name : str, default="Gradient Boosting Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                en.GradientBoostingRegressor(**kwargs),
                name
                )

    def hist_gradient_boost_regressor(
            self,
            name: str = "Histogram-Based Gradient Boosting Regression",
            **kwargs
            ):
        """
        Fit x on y using histogram-based gradient boosting regression

        Parameters
        ----------
        name : str, default="Histogram-Based Gradient Boosting Regression"
            Name of classification technique.
        -Based
            Gradient Boosting Regression
        """
        self._sklearn_regression_meta(
                en.HistGradientBoostingRegressor(**kwargs),
                name
                )

    def mlp_regressor(
            self,
            name: str = "Multi-Layer Perceptron Regression",
            **kwargs
            ):
        """
        Fit x on y using multi-layer perceptrons

        Parameters
        ----------
        name : str, default="Multi-Layer Perceptron Regression"
            Name of classification technique.
        -Layer Perceptron
            Regression
        """
        self._sklearn_regression_meta(
                nn.MLPRegressor(**kwargs),
                name
                )

    def svr(self, name: str = "Support Vector Regression", **kwargs):
        """
        Fit x on y using support vector regression

        Parameters
        ----------
        name : str, default="Support Vector Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                svm.SVR(**kwargs),
                name
                )

    def linear_svr(
            self,
            name: str = "Linear Support Vector Regression",
            **kwargs
            ):
        """
        Fit x on y using linear support vector regression

        Parameters
        ----------
        name : str, default="Linear Support Vector Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                svm.LinearSVR(**kwargs),
                name
                )

    def nu_svr(self, name: str = "Nu-Support Vector Regression", **kwargs):
        """
        Fit x on y using nu-support vector regression

        Parameters
        ----------
        name : str, default="Nu-Support Vector Regression"
            Name of classification technique.
        -Support Vector
            Regression
        """
        self._sklearn_regression_meta(
                svm.LinearSVR(**kwargs),
                name
                )

    def gaussian_process(
            self,
            name: str = "Gaussian Process Regression",
            **kwargs
            ):
        """
        Fit x on y using gaussian process regression

        Parameters
        ----------
        name : str, default="Gaussian Process Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                gp.GaussianProcessRegressor(**kwargs),
                name
                )

    def pls(self, name: str = "PLS Regression", **kwargs):
        """
        Fit x on y using pls regression

        Parameters
        ----------
        name : str, default="PLS Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                cd.PLSRegression(n_components=1, **kwargs),
                name
                )

    def isotonic(self, name: str = "Isotonic Regression", **kwargs):
        """
        Fit x on y using isotonic regression

        Parameters
        ----------
        name : str, default="Isotonic Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                iso.IsotonicRegression(**kwargs),
                name,
                max_coeffs=1
                )

    def xgboost(self, name: str = "XGBoost Regression", **kwargs):
        """
        Fit x on y using xgboost regression

        Parameters
        ----------
        name : str, default="XGBoost Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                xgb.XGBRegressor(**kwargs),
                name
                )

    def xgboost_rf(
            self,
            name: str = "XGBoost Random Forest Regression",
            **kwargs
            ):
        """
        Fit x on y using xgboosted random forest regression

        Parameters
        ----------
        name : str, default="XGBoost Random Forest Regression"
            Name of classification technique.
        """
        self._sklearn_regression_meta(
                xgb.XGBRFRegressor(**kwargs),
                name
                )

    def return_measurements(self) -> dict[str, pd.DataFrame]:
        """
        Returns the measurements used, with missing values and
        non-overlapping measurements excluded

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary with 2 keys:

            |Key|Value|
            |---|---|
            |x|`x_data`|
            |y|`y_data`|

        """
        return {
                'x': self.x_data,
                'y': self.y_data
                }

    def return_models(self) -> dict[str,  # Technique
                                    dict[str,  # Scaling method
                                         dict[str,  # Variables used
                                              dict[int,  # Fold
                                                   Pipeline]]]]:
        """
        Returns the models stored in the object

        Returns
        -------
        dict[str, str, str, int, Pipeline]
            The calibrated models. They are stored in a nested structure as
            follows:
            1. Primary Key, name of the technique (e.g Lasso Regression).
            2. Scaling technique (e.g Yeo-Johnson Transform).
            3. Combination of variables used or `target` if calibration is
            univariate (e.g "`target` + a + b).
            4. Fold, which fold was used excluded from the calibration. If data
            folds 0-3.
            if 5-fold cross validated, a key of 4 indicates the data was
            trained on
        """
        return self.models
