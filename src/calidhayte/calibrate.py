""" Contains code used to perform a range of univariate and multivariate
regressions on provided data.
>>>>>>> more_cals

Acts as a wrapper for scikit-learn [^skl], XGBoost [^xgb] and PyMC (via Bambi)
[^pmc]

[^skl]: https://scikit-learn.org/stable/modules/classes.html
[^xgb]: https://xgboost.readthedocs.io/en/stable/python/python_api.html
[^pmc]: https://bambinos.github.io/bambi/api/
"""

from collections.abc import Iterable
from copy import deepcopy as dc
from pathlib import Path
import pickle
import sys
from typing import Any, List, Literal, Optional, Union
import warnings

# import bambi as bmb
import numpy as np
import pandas as pd
from pygam import GAM, LinearGAM, ExpectileGAM
import scipy
from scipy.stats import uniform
import sklearn as skl
from sklearn import ensemble as en
from sklearn import gaussian_process as gp
from sklearn import isotonic as iso
from sklearn import linear_model as lm
from sklearn import neural_network as nn
from sklearn import svm
from sklearn import tree
from sklearn.gaussian_process import kernels as kern
import sklearn.preprocessing as pre
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb


def cont_strat_folds(
    df: pd.DataFrame,
    target_var: str,
    splits: int = 5,
    strat_groups: int = 5,
    validation_size: float = 0.1,
    seed: int = 62,
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
    validation_size : float, default = 0.1
        Size of measurements to keep aside for validation
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
    _df["Fold"] = "Validation"
    skf = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
    _df["Group"] = pd.qcut(_df.loc[:, target_var], strat_groups, labels=False)

    group_label = _df.loc[:, "Group"]

    train_set, val_set = train_test_split(
        _df,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=group_label,
    )

    group_label = train_set.loc[:, "Group"]

    for fold_number, (_, v) in enumerate(skf.split(group_label, group_label)):
        _temp_df = train_set.iloc[v, :]
        _temp_df.loc[:, "Fold"] = fold_number
        train_set.iloc[v, :] = _temp_df
    return pd.concat([train_set, val_set]).sort_index().drop("Group", axis=1)


class Calibrate:
    """
    Calibrate x against y using a range of different methods provided by
    scikit-learn[^skl], xgboost[^xgb] and PyMC (via Bambi)[^pmc].

    [^skl]: https://scikit-learn.org/stable/modules/classes.html
    [^xgb]: https://xgboost.readthedocs.io/en/stable/python/python_api.html
    [^pmc]: https://bambinos.github.io/bambi/api/

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
                    "None",
                    "Standard Scale",
                    "MinMax Scale",
                    "Yeo-Johnson Transform",
                    "Box-Cox Transform",
                    "Quantile Transform (Uniform)",
                    "Quantile Transform (Gaussian)",
                ]
            ],
            Literal[
                "All",
                "None",
                "Standard Scale",
                "MinMax Scale",
                "Yeo-Johnson Transform",
                "Box-Cox Transform",
                "Quantile Transform (Uniform)",
                "Quantile Transform (Gaussian)",
            ],
        ] = "None",
        random_search_iterations: int = 25,
        validation_size: float = 0.1,
        verbosity: int = 0,
        n_jobs: int = -1,
        pickle_path: Optional[Path] = None,
        seed: int = 62,
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
        """
        if target not in x_data.columns or target not in y_data.columns:
            raise ValueError(f"{target} does not exist in both columns.")
        join_index = (
            x_data.join(y_data, how="inner", lsuffix="x", rsuffix="y")
            .dropna()
            .index
        )
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
            "None": None,
            "Standard Scale": pre.StandardScaler(),
            "MinMax Scale": pre.MinMaxScaler(),
            "Yeo-Johnson Transform": pre.PowerTransformer(
                method="yeo-johnson"
            ),
            "Box-Cox Transform": pre.PowerTransformer(method="box-cox"),
            "Quantile Transform (Uniform)": pre.QuantileTransformer(
                output_distribution="uniform"
            ),
            "Quantile Transform (Gaussian)": pre.QuantileTransformer(
                output_distribution="normal"
            ),
        }
        """
        Keys for scaling algorithms available in the pipelines
        """
        self.scaler: list[str] = list()
        """
        The scaling algorithm(s) to preprocess the data with
        """
        self.y_data = cont_strat_folds(
            y_data.loc[join_index, :],
            target,
            folds,
            strat_groups,
            validation_size,
            seed,
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
        if isinstance(scaler, str):
            if scaler == "All":
                if bool(self.x_data.le(0).any(axis=None)) or bool(
                    self.y_data.drop("Fold", axis=1).le(0).any(axis=None)
                ):
                    self.scaler_list.pop("Box-Cox Transform")
                    warnings.warn(
                        "WARN: "
                        "Box-Cox is not compatible with provided measurements"
                    )
                self.scaler.extend(self.scaler_list.keys())
            elif scaler in self.scaler_list.keys():
                self.scaler.append(scaler)
            else:
                self.scaler.append("None")
                warnings.warn(f"Scaling algorithm {scaler} not recognised")
        elif isinstance(scaler, (tuple, list)):
            for sc in scaler:
                if sc == "Box-Cox Transform" and not any(
                    (
                        bool(self.x_data.lt(0).any(axis=None)),
                        bool(self.y_data.lt(0).any(axis=None)),
                    )
                ):
                    warnings.warn(
                        "Box-Cox is not compatible with provided measurements"
                    )
                    continue
                if sc in self.scaler_list.keys():
                    self.scaler.append(sc)
                else:
                    warnings.warn(f"Scaling algorithm {sc} not recognised")
        else:
            raise ValueError(
                "scaler parameter should be string, list or tuple"
            )
        if not self.scaler:
            warnings.warn(
                "No valid scaling algorithms provided, defaulting to None"
            )
            self.scaler.append("None")

        self.models: dict[
            str,  # Technique name
            dict[
                str,  # Scaling technique
                dict[str, dict[int, Pipeline]],  # Variable combo  # Fold
            ],
        ] = dict()
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
        self.folds: int = folds
        """
        The number of folds used in k-fold cross validation
        """
        self.rs_iter: int = random_search_iterations
        """
        Number of iterations to use in random search
        """
        self.verbosity: int = verbosity
        """
        Verbosity of output when using random search
        """
        self.n_jobs: int = n_jobs
        """
        Number of processor cores to use
        """
        self.pkl = pickle_path

    def _sklearn_regression_meta(
        self,
        reg: Union[
            skl.base.RegressorMixin,
            RandomizedSearchCV,
            GAM,
            Literal["t", "gaussian"],
        ],
        name: str,
        min_coeffs: int = 1,
        max_coeffs: int = (sys.maxsize * 2) + 1,
        random_search: bool = False,
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
        random_search : bool
            Whether RandomizedSearch is used or not

        Raises
        ------
        NotImplementedError
            PyMC currently doesn't work, TODO
        """
        x_secondary_cols = self.x_data.drop(self.target, axis=1).columns
        # All columns in x_data that aren't the target variable
        if len(x_secondary_cols) > 0:
            products = [[np.nan, col] for col in x_secondary_cols]
            secondary_vals = pd.MultiIndex.from_product(products)
        else:
            secondary_vals = [None]
        # Get all possible combinations of secondary variables in a pandas
        # MultiIndex
        if self.models.get(name) is None:
            self.models[name] = dict()
            # If the classification technique hasn't been used yet,
            # add its key to the models dictionary
        for scaler in self.scaler:
            if self.models[name].get(scaler) is None:
                self.models[name][scaler] = dict()
                # If the scaling technique hasn't been used with the
                # classification
                # technique yet, add its key to the nested dictionary
            for sec_vals in secondary_vals:
                # Loop over all combinations of secondary values
                if sec_vals is not None:
                    vals = [self.target] + [v for v in sec_vals if v == v]
                else:
                    vals = [self.target]
                vals_str = " + ".join(vals)
                if len(vals) < min_coeffs or len(vals) > max_coeffs:
                    # Skip if number of coeffs doesn't lie within acceptable
                    # range
                    # for technique. For example, isotonic regression
                    # only works with one variable
                    continue
                self.models[name][scaler][vals_str] = dict()
                #                if random_search:
                #                    pipeline = Pipeline([
                #                        ("Selector", ColumnTransformer([
                #                       ("selector", "passthrough", vals)
                #                            ], remainder="drop")
                #                         ),
                #                        ("Scaler", self.scaler_list[scaler]),
                #                        ("Regression", reg)
                #                        ])
                #                    pipeline.fit(
                #                        self.x_data,
                #                        self.y_data.loc[:, self.target]
                #                            )
                #   self.models[name][scaler][vals_str][0] = dc(pipeline)
                #                    continue
                #
                for fold in self.y_data.loc[:, "Fold"].unique():
                    if fold == "Validation":
                        continue
                    y_data = self.y_data[self.y_data.loc[:, "Fold"] != fold]
                    if reg in ["t", "gaussian"]:
                        # If using PyMC bayesian model,
                        # then store result in pipeline
                        # Currently doesn't work as PyMC models
                        # can't be pickled, so don't function with deepcopy.
                        # Needs looking into
                        raise NotImplementedError(
                            "PyMC functions currently don't work with deepcopy"
                        )
                    #                    sc = scalers[scaler]
                    #                    if sc is not None:
                    #                        x_data = sc.fit_transform(
                    #                   self.x_data.loc[y_data.index, :]
                    #                                )
                    #                    else:
                    #           x_data = self.x_data.loc[y_data.index, :]
                    #               x_data['y'] = y_data.loc[:, self.target]
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
                        pipeline = Pipeline(
                            [
                                (
                                    "Selector",
                                    ColumnTransformer(
                                        [("selector", "passthrough", vals)],
                                        remainder="drop",
                                    ),
                                ),
                                ("Scaler", self.scaler_list[scaler]),
                                ("Regression", reg),
                            ]
                        )
                        pipeline.fit(
                            self.x_data.loc[y_data.index, :],
                            y_data.loc[:, self.target],
                        )
                    if isinstance(self.pkl, Path):
                        pkl_path = self.pkl / name / scaler / vals_str
                        pkl_path.mkdir(parents=True, exist_ok=True)
                        pkl_file = pkl_path / f"{fold}.pkl"
                        with pkl_file.open("wb") as pkl:
                            pickle.dump(pipeline, pkl)
                        self.models[name][scaler][vals_str][fold] = pkl_file
                    else:
                        self.models[name][scaler][vals_str][fold] = dc(
                            pipeline
                        )

    def pymc_bayesian(
        self,
        family: Literal[
            "Gaussian",
            "Student T",
        ] = "Gaussian",
        name: str = " PyMC Bayesian",
        **kwargs,
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
        model_families: dict[str, Literal["t", "gaussian"]] = {
            "Gaussian": "gaussian",
            "Student T": "t",
        }
        self._sklearn_regression_meta(
            model_families[family], f"{name} ({model_families})", **kwargs
        )

    def linreg(
        self,
        name: str = "Linear Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {},
        **kwargs,
    ):
        """
        Fit x on y via linear regression

        Parameters
        ----------
        name : str, default="Linear Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[
                str,
                Union[
                    scipy.stats.rv_continuous,
                    List[Union[int, str, float]]
                ]
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.LinearRegression(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.LinearRegression(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def ridge(
        self,
        name: str = "Ridge Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "alpha": uniform(loc=0, scale=2),
            "tol": uniform(loc=0, scale=1),
            "solver": [
                "svd",
                "cholesky",
                "lsqr",
                "sparse_cg",
                "sag",
                "saga",
                "lbfgs",
            ],
        },
        **kwargs,
    ):
        """
        Fit x on y via ridge regression

        Parameters
        ----------
        name : str, default="Ridge Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[
                str,
                Union[
                    scipy.stats.rv_continuous,
                    List[Union[int, str, float]]
                ]
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.Ridge(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.Ridge(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def ridge_cv(
        self,
        name: str = "Ridge Regression (Cross Validated)",
        random_search: bool = False,
        **kwargs,
    ):
        """
        Fit x on y via cross-validated ridge regression.
        Already cross validated so random search not required

        Parameters
        ----------
        name : str, default="Ridge Regression (Cross Validated)"
            Name of classification technique
        random_search : bool, default=False
            Not used

        """
        _ = random_search
        self._sklearn_regression_meta(
            lm.RidgeCV(**kwargs, cv=self.folds), name, random_search=True
        )

    def lasso(
        self,
        name: str = "Lasso Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "alpha": uniform(loc=0, scale=2),
            "tol": uniform(loc=0, scale=1),
            "selection": ["cyclic", "random"],
        },
        **kwargs,
    ):
        """
        Fit x on y via lasso regression

        Parameters
        ----------
        name : str, default="Lasso Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[
                str,
                Union[
                    scipy.stats.rv_continuous,
                    List[Union[int, str, float]]
                ]
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.Lasso(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.Lasso(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def lasso_cv(
        self,
        name: str = "Lasso Regression (Cross Validated)",
        random_search: bool = False,
        **kwargs,
    ):
        """
        Fit x on y via cross-validated lasso regression.
        Already cross validated so random search not required

        Parameters
        ----------
        name : str, default="Lasso Regression (Cross Validated)"
            Name of classification technique
        random_search : bool, default=False
            Not used

        """
        _ = random_search
        self._sklearn_regression_meta(
            lm.LassoCV(**kwargs, cv=self.folds), name, random_search=True
        )

    def elastic_net(
        self,
        name: str = "Elastic Net Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "alpha": uniform(loc=0, scale=2),
            "l1_ratio": uniform(loc=0, scale=1),
            "tol": uniform(loc=0, scale=1),
            "selection": ["cyclic", "random"],
        },
        **kwargs,
    ):
        """
        Fit x on y via elastic net regression

        Parameters
        ----------
        name : str, default="Elastic Net Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[
                str,
                Union[
                    scipy.stats.rv_continuous,
                    List[Union[int, str, float]]
                ]
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.ElasticNet(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.ElasticNet(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def elastic_net_cv(
        self,
        name: str = "Elastic Net Regression (Cross Validated)",
        random_search: bool = False,
        **kwargs,
    ):
        """
        Fit x on y via cross-validated elastic regression.
        Already cross validated so random search not required

        Parameters
        ----------
        name : str, default="Lasso Regression (Cross Validated)"
            Name of classification technique
        random_search : bool, default=False
            Not used
        """
        _ = random_search
        self._sklearn_regression_meta(
            lm.ElasticNetCV(**kwargs, cv=self.folds), name, random_search=True
        )

    def lars(
        self,
        name: str = "Least Angle Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {"n_nonzero_coefs": list(range(1, 11))},
        **kwargs,
    ):
        """
        Fit x on y via least angle regression

        Parameters
        ----------
        name : str, default="Least Angle Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[
                str,
                Union[
                    scipy.stats.rv_continuous,
                    List[Union[int, str, float]]
                ]
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.Lars(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.Lars(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def lars_lasso(
        self,
        name: str = "Least Angle Lasso Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {"alpha": uniform(loc=0, scale=2)},
        **kwargs,
    ):
        """
        Fit x on y via least angle lasso regression

        Parameters
        ----------
        name : str, default="Least Angle Lasso Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[
                str,
                Union[
                    scipy.stats.rv_continuous,
                    List[Union[int, str, float]]
                ]
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.LassoLars(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.LassoLars(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def omp(
        self,
        name: str = "Orthogonal Matching Pursuit",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {"n_nonzero_coefs": list(range(1, 11))},
        **kwargs,
    ):
        """
        Fit x on y via orthogonal matching pursuit regression

        Parameters
        ----------
        name : str, default="Orthogonal Matching Pursuit"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[
                str,
                Union[
                    scipy.stats.rv_continuous,
                    List[Union[int, str, float]]
                ]
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.OrthogonalMatchingPursuit(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.OrthogonalMatchingPursuit(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
            min_coeffs=2,
        )

    def bayesian_ridge(
        self,
        name: str = "Bayesian Ridge Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "tol": uniform(loc=0, scale=1),
            "alpha_1": uniform(loc=0, scale=1),
            "alpha_2": uniform(loc=0, scale=1),
            "lambda_1": uniform(loc=0, scale=1),
            "lambda_2": uniform(loc=0, scale=1),
        },
        **kwargs,
    ):
        """
        Fit x on y via bayesian ridge regression

        Parameters
        ----------
        name : str, default="Bayesian Ridge Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[
                str,
                Union[
                    scipy.stats.rv_continuous,
                    List[Union[int, str, float]]
                ]
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.BayesianRidge(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.BayesianRidge(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def bayesian_ard(
        self,
        name: str = "Bayesian Automatic Relevance Detection",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "tol": uniform(loc=0, scale=1),
            "alpha_1": uniform(loc=0, scale=1),
            "alpha_2": uniform(loc=0, scale=1),
            "lambda_1": uniform(loc=0, scale=1),
            "lambda_2": uniform(loc=0, scale=1),
        },
        **kwargs,
    ):
        """
        Fit x on y via bayesian automatic relevance detection

        Parameters
        ----------
        name : str, default="Bayesian Automatic Relevance Detection"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[
                str,
                Union[
                    scipy.stats.rv_continuous,
                    List[Union[int, str, float]]
                ]
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.ARDRegression(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.ARDRegression(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def tweedie(
        self,
        name: str = "Tweedie Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "power": [0, 1, 1.5, 2, 2.5, 3],
            "alpha": uniform(loc=0, scale=2),
            "solver": ["lbfgs", "newton-cholesky"],
            "tol": uniform(loc=0, scale=1),
        },
        **kwargs,
    ):
        """
        Fit x on y via tweedie regression

        Parameters
        ----------
        name : str, default="Tweedie Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.TweedieRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.TweedieRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def stochastic_gradient_descent(
        self,
        name: str = "Stochastic Gradient Descent",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "tol": uniform(loc=0, scale=1),
            "loss": [
                "squared_error",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ],
            "penalty": ["l2", "l1", "elasticnet", None],
            "alpha": uniform(loc=0, scale=0.001),
            "l1_ratio": uniform(loc=0, scale=1),
            "epsilon": uniform(loc=0, scale=1),
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "eta0": uniform(loc=0, scale=0.1),
            "power_t": uniform(loc=0, scale=1),
        },
        **kwargs,
    ):
        """
        Fit x on y via stochastic gradient descent

        Parameters
        ----------
        name : str, default="Stochastic Gradient Descent"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[
                str,
                Union[
                    scipy.stats.rv_continuous,
                    List[Union[int, str, float]]
                ]
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.SGDRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.SGDRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def passive_aggressive(
        self,
        name: str = "Passive Aggressive Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "C": uniform(loc=0, scale=2),
            "tol": uniform(loc=0, scale=1),
            "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
            "epsilon": uniform(loc=0, scale=1),
        },
        **kwargs,
    ):
        """
        Fit x on y via passive aggressive regression

        Parameters
        ----------
        name : str, default="Passive Aggressive Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.PassiveAggressiveRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.PassiveAggressiveRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def ransac(
        self,
        name: str = "RANSAC",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "estimator": [
                lm.LinearRegression(),
                lm.TheilSenRegressor(),
                lm.LassoLarsCV(),
            ],
            "min_samples": [1e-4, 1e-3, 1e-2],
        },
        **kwargs,
    ):
        """
        Fit x on y via ransac

        Parameters
        ----------
        name : str, default="RANSAC"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.RANSACRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.RANSACRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def theil_sen(
        self,
        name: str = "Theil-Sen Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {"tol": uniform(loc=0, scale=1)},
        **kwargs,
    ):
        """
        Fit x on y via theil-sen regression

        Parameters
        ----------
        name : str, default="Theil-Sen Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.TheilSenRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.TheilSenRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def huber(
        self,
        name: str = "Huber Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "epsilon": uniform(loc=1, scale=4),
            "alpha": uniform(loc=0, scale=0.01),
            "tol": uniform(loc=0, scale=1),
        },
        **kwargs,
    ):
        """
        Fit x on y via huber regression

        Parameters
        ----------
        name : str, default="Huber Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.HuberRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.HuberRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def quantile(
        self,
        name: str = "Quantile Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "quantile": uniform(loc=0, scale=2),
            "alpha": uniform(loc=0, scale=2),
            "solver": [
                "highs-ds",
                "highs-ipm",
                "highs",
                "revised simplex",
            ],
        },
        **kwargs,
    ):
        """
        Fit x on y via quantile regression

        Parameters
                'interior-point',
        ----------
        name : str, default="Quantile Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                lm.QuantileRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = lm.QuantileRegressor(solver="highs", **kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def decision_tree(
        self,
        name: str = "Decision Tree",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "criterion": [
                "squared_error",
                "friedman_mse",
                "absolute_error",
                "poisson",
            ],
            "splitter": ["best", "random"],
            "max_features": [None, "sqrt", "log2"],
            "ccp_alpha": uniform(loc=0, scale=2),
        },
        **kwargs,
    ):
        """
        Fit x on y via decision tree

        Parameters
        ----------
        name : str, default="Decision Tree"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                tree.DecisionTreeRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = tree.DecisionTreeRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def extra_tree(
        self,
        name: str = "Extra Tree",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "criterion": [
                "squared_error",
                "friedman_mse",
                "absolute_error",
                "poisson",
            ],
            "splitter": ["best", "random"],
            "max_features": [None, "sqrt", "log2"],
            "ccp_alpha": uniform(loc=0, scale=2),
        },
        **kwargs,
    ):
        """
        Fit x on y via extra tree

        Parameters
        ----------
        name : str, default="Extra Tree"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                tree.ExtraTreeRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = tree.ExtraTreeRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def random_forest(
        self,
        name: str = "Random Forest",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "n_estimators": [5, 25, 100, 250],
            "max_samples": uniform(loc=0.01, scale=0.99),
            "criterion": [
                "squared_error",
                "friedman_mse",
                "absolute_error",
                "poisson",
            ],
            "max_features": [None, "sqrt", "log2"],
            "ccp_alpha": uniform(loc=0, scale=2),
        },
        **kwargs,
    ):
        """
        Fit x on y via random forest

        Parameters
        ----------
        name : str, default="Random Forest"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                en.RandomForestRegressor(bootstrap=True, **kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = en.RandomForestRegressor(bootstrap=True, **kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def extra_trees_ensemble(
        self,
        name: str = "Extra Trees Ensemble",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "n_estimators": [5, 25, 100, 250],
            "max_samples": uniform(loc=0.01, scale=0.99),
            "criterion": [
                "squared_error",
                "friedman_mse",
                "absolute_error",
                "poisson",
            ],
            "max_features": [None, "sqrt", "log2"],
            "ccp_alpha": uniform(loc=0, scale=2),
        },
        **kwargs,
    ):
        """
        Fit x on y via extra trees ensemble

        Parameters
        ----------
        name : str, default="Extra Trees Ensemble"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                en.ExtraTreesRegressor(bootstrap=True, **kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = en.ExtraTreesRegressor(bootstrap=True, **kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def gradient_boost_regressor(
        self,
        name: str = "Gradient Boosting Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "loss": ["squared_error", "absolute_error", "huber", "quantile"],
            "learning_rate": uniform(loc=0, scale=2),
            "n_estimators": [5, 25, 100, 250],
            "subsample": uniform(loc=0.01, scale=0.99),
            "criterion": ["friedman_mse", "squared_error"],
            "max_features": [None, "sqrt", "log2"],
            "init": [None, "zero", lm.LinearRegression, lm.TheilSenRegressor],
            "ccp_alpha": uniform(loc=0, scale=2),
        },
        **kwargs,
    ):
        """
        Fit x on y via gradient boosting regression

        Parameters
        ----------
        name : str, default="Gradient Boosting Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                en.GradientBoostingRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = en.GradientBoostingRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def hist_gradient_boost_regressor(
        self,
        name: str = "Histogram-Based Gradient Boosting Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "loss": [
                "squared_error",
                "absolute_error",
                "gamma",
                "poisson",
                "quantile",
            ],
            "quantile": uniform(loc=0, scale=1),
            "learning_rate": uniform(loc=0, scale=2),
            "max_iter": [5, 10, 25, 50, 100, 200, 250, 500],
            "l2_regularization": uniform(loc=0, scale=2),
            "max_bins": [1, 3, 7, 15, 31, 63, 127, 255],
        },
        **kwargs,
    ):
        """
        Fit x on y via histogram-based gradient boosting regression

        Parameters
        ----------
        name : str, default="Histogram-Based Gradient Boosting Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                en.HistGradientBoostingRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = en.HistGradientBoostingRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def mlp_regressor(
        self,
        name: str = "Multi-Layer Perceptron Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "hidden_layer_sizes": [
                (100,),
                (100, 200),
                (10,),
                (200, 400),
                (100, 200, 300),
            ],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": uniform(loc=0, scale=0.1),
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": uniform(loc=0, scale=0.1),
            "power_t": uniform(loc=0.1, scale=0.9),
            "max_iter": [5, 10, 25, 50, 100, 200, 250, 500],
            "shuffle": [True, False],
            "momentum": uniform(loc=0.1, scale=0.9),
            "beta_1": uniform(loc=0.1, scale=0.9),
            "beta_2": uniform(loc=0.1, scale=0.9),
            "epsilon": uniform(loc=1e-8, scale=1e-6),
        },
        **kwargs,
    ):
        """
        Fit x on y via multi-layer perceptron regression

        Parameters
        ----------
        name : str, default="Multi-Layer Perceptron Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                nn.MLPRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = nn.MLPRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def svr(
        self,
        name: str = "Support Vector Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "kernel": [
                "linear",
                "poly",
                "rbf",
                "sigmoid",
            ],
            "degree": [2, 3, 4],
            "gamma": ["scale", "auto"],
            "coef0": uniform(loc=0, scale=1),
            "C": uniform(loc=0.1, scale=1.9),
            "epsilon": uniform(loc=1e-8, scale=1),
            "shrinking": [True, False],
        },
        **kwargs,
    ):
        """
        Fit x on y via support vector regression

        Parameters
        ----------
        name : str, default="Support Vector Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                svm.SVR(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = svm.SVR(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def linear_svr(
        self,
        name: str = "Linear Support Vector Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "C": uniform(loc=0.1, scale=1.9),
            "epsilon": uniform(loc=1e-8, scale=1),
            "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        },
        **kwargs,
    ):
        """
        Fit x on y via linear support vector regression

        Parameters
        ----------
        name : str, default="Linear Support Vector Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                svm.LinearSVR(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = svm.LinearSVR(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def nu_svr(
        self,
        name: str = "Nu-Support Vector Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "kernel": [
                "linear",
                "poly",
                "rbf",
                "sigmoid",
            ],
            "degree": [2, 3, 4],
            "gamma": ["scale", "auto"],
            "coef0": uniform(loc=0, scale=1),
            "shrinking": [True, False],
            "nu": uniform(loc=0, scale=1),
        },
        **kwargs,
    ):
        """
        Fit x on y via nu-support vector regression

        Parameters
        ----------
        name : str, default="Nu-Support Vector Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                svm.NuSVR(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = svm.NuSVR(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def gaussian_process(
        self,
        name: str = "Gaussian Process Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "kernel": [
                None,
                kern.RBF,
                kern.Matern,
                kern.DotProduct,
                kern.WhiteKernel,
                kern.CompoundKernel,
                kern.ExpSineSquared,
            ],
            "alpha": uniform(loc=0, scale=1e-8),
            "normalize_y": [True, False],
        },
        **kwargs,
    ):
        """
        Fit x on y via gaussian process regression

        Parameters
        ----------
        name : str, default="Gaussian Process Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                gp.GaussianProcessRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = gp.GaussianProcessRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def isotonic(
        self,
        name: str = "Isotonic Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {"increasing": [True, False]},
        **kwargs,
    ):
        """
        Fit x on y via isotonic regression

        Parameters
        ----------
        name : str, default="Isotonic Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                iso.IsotonicRegression(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = iso.IsotonicRegression(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
            max_coeffs=1,
        )

    def xgboost(
        self,
        name: str = "XGBoost Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "n_estimators": [5, 25, 100, 250],
            "max_bins": [1, 3, 7, 15, 31, 63, 127, 255],
            "grow_policy": ["depthwise", "lossguide"],
            "learning_rate": uniform(loc=0, scale=2),
            "tree_method": ["exact", "approx", "hist"],
            "gamma": uniform(loc=0, scale=1),
            "subsample": uniform(loc=0, scale=1),
            "reg_alpha": uniform(loc=0, scale=1),
            "reg_lambda": uniform(loc=0, scale=1),
        },
        **kwargs,
    ):
        """
        Fit x on y via xgboost regression

        Parameters
        ----------
        name : str, default="XGBoost Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                xgb.XGBRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = xgb.XGBRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def xgboost_rf(
        self,
        name: str = "XGBoost Random Forest Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "n_estimators": [5, 25, 100, 250],
            "max_bin": [1, 3, 7, 15, 31, 63, 127, 255],
            "grow_policy": ["depthwise", "lossguide"],
            "learning_rate": uniform(loc=0, scale=2),
            "tree_method": ["exact", "approx", "hist"],
            "gamma": uniform(loc=0, scale=1),
            "subsample": uniform(loc=0, scale=1),
            "reg_alpha": uniform(loc=0, scale=1),
            "reg_lambda": uniform(loc=0, scale=1),
        },
        **kwargs,
    ):
        """
        Fit x on y via xgboosted random forest regression

        Parameters
        ----------
        name : str, default="XGBoost Random Forest Regression"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                xgb.XGBRFRegressor(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = xgb.XGBRFRegressor(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def linear_gam(
        self,
        name: str = "Linear GAM",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "max_iter": [100, 500, 1000],
            "callbacks": ['deviance', 'diffs']
        },
        **kwargs,
    ):
        """
        Fit x on y via a linear GAM

        Parameters
        ----------
        name : str, default="Linear GAM"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                LinearGAM(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = LinearGAM(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
        )

    def expectile_gam(
        self,
        name: str = "Expectile GAM",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, List[Union[int, str, float]]]
        ] = {
            "max_iter": [100, 500, 1000],
            "callbacks": ['deviance', 'diffs'],
            "expectile": uniform(loc=0, scale=1)
        },
        **kwargs,
    ):
        """
        Fit x on y via an expectile GAM

        Parameters
        ----------
        name : str, default="Expectile GAM"
            Name of classification technique.
        random_search : bool, default=False
            Whether to perform RandomizedSearch to optimise parameters
        parameters : dict[\
                str,\
                Union[\
                    scipy.stats.rv_continuous,\
                    List[Union[int, str, float]]\
                ]\
            ], default=Preset distributions
            The parameters used in RandomizedSearchCV
        """
        if random_search:
            classifier = RandomizedSearchCV(
                ExpectileGAM(**kwargs),
                parameters,
                n_iter=self.rs_iter,
                verbose=self.verbosity,
                n_jobs=self.n_jobs,
                cv=self.folds,
            )
        else:
            classifier = ExpectileGAM(**kwargs)
        self._sklearn_regression_meta(
            classifier,
            f'{name}{" (Random Search)" if random_search else ""}',
            random_search=random_search,
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
        return {"x": self.x_data, "y": self.y_data}

    def return_models(
        self,
    ) -> dict[
        str,  # Technique
        dict[
            str,  # Scaling method
            dict[str, dict[int, Pipeline]],  # Variables used  # Fold
        ],
    ]:
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

    def clear_models(self):
        """ """
        del self.models
        self.models = dict()
