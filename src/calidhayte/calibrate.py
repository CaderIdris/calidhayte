"""Perform multiple regressions on dataframes.

Acts as a wrapper for scikit-learn [^skl], XGBoost [^xgb] and PyMC (via Bambi)
[^pmc]

[^skl]: https://scikit-learn.org/stable/modules/classes.html
[^xgb]: https://xgboost.readthedocs.io/en/stable/python/python_api.html
[^pmc]: https://bambinos.github.io/bambi/api/
"""

from collections.abc import Iterable
from copy import deepcopy as dc
import logging
from pathlib import Path
import pickle
import sys
from typing import (
    ClassVar,
    Type,
    Literal,
    Optional,
    TypeVar,
    TypeAlias,
    Union,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

# import bambi as bmb
import pandas as pd
from pygam import GAM, LinearGAM, ExpectileGAM
import scipy
from scipy.stats import uniform
from sklearn import ensemble as en
from sklearn import gaussian_process as gp
from sklearn import isotonic as iso
from sklearn import linear_model as lm
from sklearn import neural_network as nn
from sklearn import svm
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.gaussian_process import kernels as kern
import sklearn.preprocessing as pre
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb

logger = logging.getLogger(f"__main__.{__name__}")

ScalingOptions: TypeAlias = Union[
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
]

CalibrateType = TypeVar("CalibrateType", bound="Calibrate")


class VIF(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self: Self, target: str, bound: float) -> None:
        self.target: str = target
        self.bound: float = bound
        self.columns_to_drop: list[str] = []
        self.features: list[str] = []

    def fit(self: Self, x: pd.DataFrame, y: None = None) -> Self:
        """ """
        for _ in range(x.shape[1] - 2):
            x_new = x.copy(deep=True).drop(
                columns=[self.target, *self.columns_to_drop]
            )
            vif_data = pd.Series()
            for i, col in enumerate(x_new.columns):
                vif_data[col] = variance_inflation_factor(x_new.values, i)
            if not (vif_data > self.bound).any():
                break
            largest = str(vif_data.idxmax())
            self.columns_to_drop.append(largest)
        self.features = list(
            x.copy().drop(columns=self.columns_to_drop).columns
        )
        return self

    def transform(self: Self, x: pd.DataFrame, y: None = None) -> pd.DataFrame:
        """ """
        return x.copy().drop(columns=self.columns_to_drop)

    def get_feature_names_out(
        self: Self, input_features: list[str]
    ) -> list[str]:
        """ """
        return self.features


class TimeColumnTransformer(BaseEstimator, TransformerMixin):
    """ """
    def __init__(self: Self) -> None:
        """"""
        self.earliest: Optional[int] = None
        self.latest: Optional[int] = None
        self.feature_names_in_: Optional[list[str]] = None

    def fit(self: Self, x: pd.DataFrame, _: None = None) -> Self:
        """"""
        dates = self.pandas_to_unix(pd.Series(x.index))
        self.earliest = dates.min()
        self.latest = dates.max()
        self.feature_names_in_ = list(x.columns)
        return self

    def transform(self: Self, x: pd.DataFrame, _: None = None) -> pd.DataFrame:
        """"""
        if self.earliest is None or self.latest is None:
            err = "Transformer not fitted"
            raise ValueError(err)
        dates = self.pandas_to_unix(pd.Series(x.index))
        dates.index = x.index
        new_x = x.copy()
        new_x['Time Since Origin'] = dates.sub(
            self.earliest
        ).div(
            self.latest - self.earliest
        )
        return new_x

    def get_feature_names_out(self: Self, _: list[str]) -> list[str]:
        """"""
        if self.feature_names_in_ is None:
            err = "Transformer not fitted"
            raise ValueError(err)
        return [*self.feature_names_in_, "Time Since Origin"]

    @staticmethod
    def pandas_to_unix(dates: pd.Series) -> pd.Series:
        """"""
        timezone = dates.dt.tz
        return dates.sub(
            pd.Timestamp(
                "1970-01-02",
                tz=timezone
            )
        ) // pd.Timedelta("1s")


class PolynomialPandas(BaseEstimator, TransformerMixin):
    """ """

    def __init__(
        self: Self,
        degree: int,
        selected_features: Optional[list[str]] = None
    ) -> None:
        self.degree = degree
        self.estimator = pre.PolynomialFeatures(
            degree=degree, include_bias=False
        )
        self.selected_features = selected_features

    def fit(self: Self, x: pd.DataFrame, y: None = None) -> Optional[Self]:
        """ """
        if self.selected_features is None:
            self.estimator.fit(x)
        else:
            selected_features = [
                col
                for col in x.columns
                if col in self.selected_features
            ]
            if selected_features:
                self.estimator.fit(x.loc[:, selected_features])
            else:
                return None
        return self


    def transform(self: Self, x: pd.DataFrame, y: None = None) -> pd.DataFrame:
        """ """
        if self.selected_features is None:
            transformed_array = self.estimator.transform(x)
            x = pd.DataFrame(
                transformed_array,
                index=x.index,
                columns=self.estimator.get_feature_names_out(),
            )
        else:
            selected_features = [
                col
                for col in x.columns
                if col in self.selected_features
            ]
            if selected_features:
                transformed_array = pd.DataFrame(
                        self.estimator.transform(x.loc[:, selected_features]),
                        index=x.index,
                        columns=pd.Index(selected_features)
                )
                x.loc[:, transformed_array.columns] = transformed_array
        return x

    def get_feature_names_out(
        self: Self, input_features: list[str]
    ) -> list[str]:
        """ """
        return list(self.estimator.get_feature_names_out())


class Calibrate:
    """Calibrate x against y.

    Calibrate using a range of different methods provided by
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
    >>> calibration = Calibrate.setup(
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

    scaler_list: ClassVar = {
        "None": None,
        "Standard Scale": pre.StandardScaler(),
        "MinMax Scale": pre.MinMaxScaler(),
        "Yeo-Johnson Transform": pre.PowerTransformer(method="yeo-johnson"),
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

    @classmethod
    def setup(
        cls: Type[CalibrateType],
        x_data: pd.DataFrame,
        y_data: pd.DataFrame,
        target: str,
        *,
        scaler: ScalingOptions = "None",
        interaction_degree: int = 0,
        interaction_features: Optional[list[str]] = None,
        vif_bound: Optional[float] = None,
        add_time_column: bool = False,
        pickle_path: Optional[Path] = None,
        subsample_data: Optional[Union[int, float]] = None,
        folds: int = 5,
        validation_size: float = 0.1,
        strat_groups: int = 10,
        seed: int = 62,
        random_search_iterations: int = 25,
        n_jobs: int = -1,
        verbosity: int = 0,
    ) -> CalibrateType:
        """Set up class with all required parameters, performs type checking.

        Parameters
        ----------
        x_data : pd.DataFrame
            Data to be calibrated.
        y_data : pd.DataFrame
            'True' data to calibrate against.
        target : str
            Column name of the primary feature to use in calibration, must be
            the name of a column in both `x_data` and `y_data`.
        scaler : ScalingOptions, default='None'
            The scaling/transform method (or list of methods) to apply to the
            data
        interaction_degree : int, default=0
            Polynomial degree for interaction variables
        interaction_features : list[str], optional
            Features to transform with the PolynomialFeatures transformer.
            Will use all features if not None
        vif_bound : float, optional, default=None
            Bound for vif dimensional reduction, any feature with a vif above
            the set value will be discarded
        add_time_column : bool, default=False
            Should a time column be added to the data during transformation.
            Earliest timestamp in fitted data is 0, latest timestamp is 1
        pickle_path : Path, optional, default=None
            Where to save trained estimators as pickle files, will not save if
            path not provided
        subsample_data : float, int, optional, default=None
            Subsample of data to use
        folds : int, default=5
            Number of folds to split the data into, using stratified k-fold.
        validation_size : float, default=0.1
            Proportion of data to use as validation set
        strat_groups : int, default=10
            Number of groups to stratify against, the data will be split into
            n equally sized bins where n is the value of `strat_groups`.
        seed : int, default=62
            Random state to use when shuffling and splitting the data into n
            folds. Ensures repeatability.
        random_search_iterations : int, default=25
            Number of iterations in random search
        n_jobs : int, default=-1
            Number of jobs to run in random search
        verbosity : int, default=0
            Verbosity of random search output

        Raises
        ------
        ValueError
            Raised if the target variables (e.g. 'NO2') is not a column name in
            both dataframes.
            Raised if `scaler` is not str, tuple or list
        """
        if target not in x_data.columns or target not in y_data.columns:
            error_string = f"{target} does not exist in both columns."
            raise ValueError(error_string)
        if subsample_data is not None:
            try:
                x_data = cls.subsample_df(
                    x_data,
                    target,
                    subsample_data
                )
            except ValueError:
                logger.warning('Subset size larger than dataset size')
        join_index = (
            x_data.join(y_data, how="inner", lsuffix="x", rsuffix="y")
            .dropna()
            .index
        )
        x_data_filtered: pd.DataFrame = x_data.loc[join_index, :]
        y_data_filtered = cls.cont_strat_folds(
            y_data.loc[join_index, :],
            target,
            folds,
            strat_groups,
            validation_size,
            seed,
        )
        scaler_list = cls.configure_scalers(
            scaler, x_data=x_data_filtered, y_data=y_data_filtered
        )
        return cls(
            x_data=x_data_filtered,
            y_data=y_data_filtered,
            target=target,
            scalers=scaler_list,
            interaction_degree=interaction_degree,
            interaction_features=interaction_features,
            vif_bound=vif_bound,
            add_time_column=add_time_column,
            pickle_path=pickle_path,
            seed=seed,
            folds=folds,
            random_search_iterations=random_search_iterations,
            n_jobs=n_jobs,
            verbosity=verbosity,
        )

    def __init__(
        self: Self,
        *,
        x_data: pd.DataFrame,
        y_data: pd.DataFrame,
        target: str,
        scalers: list[str],
        interaction_degree: int,
        interaction_features: Optional[list[str]],
        vif_bound: Optional[float],
        add_time_column: bool,
        pickle_path: Optional[Path],
        seed: int,
        folds: int,
        random_search_iterations: int,
        n_jobs: int,
        verbosity: int,
    ) -> None:
        """Initialise class.

        Used to compare one set of measurements against another.
        It can perform both univariate and multivariate regression, though
        some techniques can only do one or the other. Multivariate regression
        can only be performed when secondary variables are provided.
        """
        self.x_data: pd.DataFrame = x_data
        """
        The data to be calibrated.
        """
        self.y_data = y_data
        """
        The data that `x_data` will be calibrated against. A '*Fold*'
        column is added using the `const_strat_folds` function which splits
        the data into k stratified folds (where k is the value of
        `folds`). It splits the continuous measurements into n bins (where n
        is the value of `strat_groups`) and distributes each bin equally
        across all folds. This significantly reduces the chances of one fold
        containing a skewed distribution relative to the whole dataset.
        """
        self.target: str = target
        """
        The name of the column in both `x_data` and `y_data` that
        will be used as the x and y variables in the calibration.
        """
        self.scaler: list[str] = scalers
        """
        The scaling algorithm(s) to preprocess the data with
        """
        self.interaction_degree: int = interaction_degree
        """
        The polynomial degree for interaction variables, will disable
        polynomial features if less than 2
        """
        self.interaction_features: Optional[list[str]] = interaction_features
        """
        The features to transform with the PolynomialFeatures transformer.
        Will use all features if None.
        """
        self.vif_bound: Optional[float] = vif_bound
        """
        Bound to use for VIF dimensional reduction, VIF not used if None
        """
        self.add_time_column: tuple[bool, ...] = (
                (True, False) if add_time_column else (False,)
        )
        """
        Add a column representing time since first timestamp seen during
        fitting pipeline to the transformation steps?
        """
        self.models: dict[
            str,  # Technique name
            dict[
                str,  # Scaling technique
                dict[
                    str, dict[int, Union[Pipeline, Path]]
                ],  # Variable combo  # Fold
            ],
        ] = {}
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
        """
        Path to save pickle files to, or None if pipelines are stored in memory
        """
        self.seed: int = seed
        """
        Seed to use for random number generator
        """
        self.folds = folds
        """
        Number of folds
        """
        self.transformer_pipelines: dict[
            str, dict[str, dict[int, Pipeline]]
        ] = {}

    def _pretrain_transformer_pipeline(self: Self) -> None:
        """Pretrain transformers."""
        x_secondary_cols = self.x_data.drop(self.target, axis=1).columns
        # All columns in x_data that aren't the target variable
        if len(x_secondary_cols) > 0:
            secondary_vals = [None, list(x_secondary_cols)]
        else:
            secondary_vals = [None]
        # Get all possible combinations of secondary variables in a pandas
        # MultiIndex
        for scaler in self.scaler:
            if self.transformer_pipelines.get(scaler) is None:
                self.transformer_pipelines[scaler] = {}
                # If the scaling technique hasn't been used with the
                # classification
                # technique yet, add its key to the nested dictionary
            for sec_vals in secondary_vals:
                for add_time_column in self.add_time_column:
                # Loop over all combinations of secondary values
                    if sec_vals is not None:
                        vals = [self.target] + [v for v in sec_vals if v == v]
                    else:
                        vals = [self.target]
                    for fold in self.y_data.loc[:, "Fold"].unique():
                        if fold == "Validation":
                            continue
                        y_data = self.y_data[
                            ~self.y_data.loc[:, "Fold"].isin(
                                [fold, "Validation"]
                            )
                        ]
                        if self.interaction_degree > 1:
                            interaction_vars = PolynomialPandas(
                                degree=self.interaction_degree,
                                selected_features=self.interaction_features

                            ).set_output(transform="pandas")
                        else:
                            interaction_vars = None

                        vif = (
                            VIF(target=self.target, bound=self.vif_bound)
                            if self.vif_bound is not None
                            else None
                        )

                        time_column = (
                                TimeColumnTransformer() if add_time_column
                                else None
                        )

                        scaler_transformer = (
                            None if self.scaler_list[scaler] is None
                            else self.scaler_list[scaler].set_output(
                                transform="pandas"
                            )
                        )

                        pipeline = Pipeline(
                            [
                                (
                                    "Selector",
                                    ColumnTransformer(
                                        [("selector", "passthrough", vals)],
                                        remainder="drop",
                                        verbose_feature_names_out=False,
                                    ).set_output(transform="pandas"),
                                ),
                                ("Interaction Variables", interaction_vars),
                                ("VIF", vif),
                                ("Scaler", scaler_transformer),
                                ("Time Column", time_column),
                            ]
                        )
                        pipeline.fit(
                            self.x_data.loc[y_data.index, :],
                            y_data.loc[:, self.target],
                        )
                        vals_str = " + ".join(
                            list(pipeline.get_feature_names_out())
                        )
                        if vals_str not in self.transformer_pipelines[scaler]:
                            self.transformer_pipelines[scaler][vals_str] = {}
                        self.transformer_pipelines[scaler][vals_str][fold] = (
                            dc(
                                pipeline
                            )
                        )

    def _sklearn_regression_meta(
        self: Self,
        reg: Union[RegressorMixin, RandomizedSearchCV, GAM],
        name: str,
        min_coeffs: int = 1,
        max_coeffs: int = (sys.maxsize * 2) + 1,
        random_search: Optional[bool] = None,
    ) -> None:
        """Format data and use sklearn classifier to fit x to y.

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
        """
        if not self.transformer_pipelines:
            self._pretrain_transformer_pipeline()
        if self.models.get(name) is None:
            self.models[name] = {}
            # If the classification technique hasn't been used yet,
            # add its key to the models dictionary
        for scaler, var_pipes in self.transformer_pipelines.items():
            if self.models[name].get(scaler) is None:
                self.models[name][scaler] = {}
                # If the scaling technique hasn't been used with the
                # classification
                # technique yet, add its key to the nested dictionary
            for sec_vars, fold_pipes in var_pipes.items():
                # Loop over all combinations of secondary values
                vals = sec_vars.split(" + ")
                if len(vals) < min_coeffs or len(vals) > max_coeffs:
                    # Skip if number of coeffs doesn't lie within acceptable
                    # range
                    # for technique. For example, isotonic regression
                    # only works with one variable
                    continue
                self.models[name][scaler][sec_vars] = {}
                for fold, pretrained_pipe in fold_pipes.items():
                    y_data = self.y_data[
                        ~self.y_data.loc[:, "Fold"].isin([fold, "Validation"])
                    ]
                    logger.debug(
                        "%s, %s, %s and %s", name, scaler, sec_vars, fold
                    )

                    pipeline = dc(pretrained_pipe)
                    transformed_data = pipeline.fit_transform(
                        self.x_data.loc[y_data.index, :]
                    )
                    reg_dc = dc(reg)
                    regressor = reg_dc.fit(
                        transformed_data, y_data.loc[:, self.target]
                    )
                    pipeline.steps.append(("Regressor", dc(regressor)))
                    if isinstance(self.pkl, Path):
                        pkl_path = self.pkl / name / scaler / sec_vars
                        pkl_path.mkdir(parents=True, exist_ok=True)
                        pkl_file = pkl_path / f"{fold}.pkl"
                        with pkl_file.open("wb") as pkl:
                            pickle.dump(pipeline, pkl)
                        self.models[name][scaler][sec_vars][fold] = pkl_file
                    else:
                        self.models[name][scaler][sec_vars][fold] = dc(
                            pipeline
                        )

    def linreg(
        self: Self,
        name: str = "Linear Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
        ] = {},
        **kwargs,
    ) -> None:
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
                    list[Union[int, str, float]]
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
        self: Self,
        name: str = "Ridge Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]
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
        self: Self,
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
        self: Self,
        name: str = "Lasso Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]
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
        self: Self,
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
        self: Self,
        name: str = "Elastic Net Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]
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
        self: Self,
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
        self: Self,
        name: str = "Least Angle Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]
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
        self: Self,
        name: str = "Least Angle Lasso Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]
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
        self: Self,
        name: str = "Orthogonal Matching Pursuit",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]
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
        self: Self,
        name: str = "Bayesian Ridge Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]
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
        self: Self,
        name: str = "Bayesian Automatic Relevance Detection",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]
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
        self: Self,
        name: str = "Tweedie Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Stochastic Gradient Descent",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]
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
        self: Self,
        name: str = "Passive Aggressive Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "RANSAC",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Theil-Sen Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Huber Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Quantile Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Decision Tree",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Extra Tree",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Random Forest",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Extra Trees Ensemble",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Gradient Boosting Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Histogram-Based Gradient Boosting Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Multi-Layer Perceptron Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Support Vector Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Linear Support Vector Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Nu-Support Vector Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Gaussian Process Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Isotonic Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "XGBoost Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "XGBoost Random Forest Regression",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Linear GAM",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
        ] = {"max_iter": [100, 500, 1000], "callbacks": ["deviance", "diffs"]},
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
                    list[Union[int, str, float]]\
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
        self: Self,
        name: str = "Expectile GAM",
        random_search: bool = False,
        parameters: dict[
            str, Union[scipy.stats.rv_continuous, list[Union[int, str, float]]]
        ] = {
            "max_iter": [100, 500, 1000],
            "callbacks": ["deviance", "diffs"],
            "expectile": uniform(loc=0, scale=1),
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
                    list[Union[int, str, float]]\
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

    @classmethod
    def configure_scalers(
        cls,
        scaler_options: ScalingOptions,
        x_data: pd.DataFrame,
        y_data: pd.DataFrame,
    ) -> list[str]:
        """Configure the scaling algorithms to use.

        Parameters
        ----------
        scaler_options : ScalingOptions
            The options chosen
        x_data : pd.DataFrame
            The data to be calibrated
        y_data : pd.DataFrame
            The data to calibrate against
        """
        scaler_list = cls.scaler_list
        scalers = []
        if isinstance(scaler_options, str):
            if scaler_options == "All":
                if (
                        bool(x_data.le(0).any(axis=None)) or
                        bool(y_data.drop("Fold", axis=1).le(0).any(axis=None)
                    ) and "Box-Cox Transform" in scaler_list
                ):
                    scaler_list.pop("Box-Cox Transform", None)
                    logger.warning(
                        "WARN: "
                        "Box-Cox is not compatible with provided measurements"
                    )
                scalers.extend(scaler_list.keys())
            elif scaler_options in scaler_list:
                scalers.append(scaler_options)
            else:
                scalers.append("None")
                logger.warning(
                    "Scaling algorithm %s not recognised", scaler_options
                )
        elif isinstance(scaler_options, (tuple, list)):
            for sc in scaler_options:
                if sc == "Box-Cox Transform" and not any(
                    (
                        bool(x_data.lt(0).any(axis=None)),
                        bool(y_data.lt(0).any(axis=None)),
                    )
                ):
                    logger.warning(
                        "Box-Cox is not compatible with provided measurements"
                    )
                    continue
                if sc in scaler_list:
                    scalers.append(sc)
                else:
                    logger.warning("Scaling algorithm %s not recognised", sc)
        else:
            scaler_error = "Scaler parameter should be string, list or tuple"
            raise TypeError(scaler_error)
        if not scalers:
            logger.warning(
                "No valid scaling algorithms provided, defaulting to None"
            )
            scalers.append("None")
        return scalers

    @staticmethod
    def subsample_df(
        df: pd.DataFrame,
        target_var: str,
        subsample_size: Union[float, int] = 1.0,
        strat_groups: int = 25,
        seed: int = 62,
    ) -> pd.DataFrame:
        """Create stratified k-folds on continuous variable.
        """
        _df = df.copy().dropna(subset=target_var)
        _df["Group"] = pd.qcut(
            _df.loc[:, target_var], strat_groups, labels=False
        )

        group_label = _df.loc[:, "Group"]

        _, subset = train_test_split(
            _df,
            test_size=subsample_size,
            random_state=seed,
            shuffle=True,
            stratify=group_label,
        )

        return (
            subset.drop("Group", axis=1)
        )

    @staticmethod
    def cont_strat_folds(
        df: pd.DataFrame,
        target_var: str,
        splits: int = 5,
        strat_groups: int = 5,
        validation_size: float = 0.1,
        seed: int = 62,
    ) -> pd.DataFrame:
        """Create stratified k-folds on continuous variable.

        Parameters
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
        _df["Group"] = pd.qcut(
            _df.loc[:, target_var], strat_groups, labels=False
        )

        group_label = _df.loc[:, "Group"]

        train_set, val_set = train_test_split(
            _df,
            test_size=validation_size,
            random_state=seed,
            shuffle=True,
            stratify=group_label,
        )

        group_label = train_set.loc[:, "Group"]

        for fold_number, (_, v) in enumerate(
            skf.split(group_label, group_label)
        ):
            _temp_df = train_set.iloc[v, :]
            _temp_df.loc[:, "Fold"] = fold_number
            train_set.iloc[v, :] = _temp_df
        return (
            pd.concat([train_set, val_set]).sort_index().drop("Group", axis=1)
        )

    def return_measurements(self: Self) -> dict[str, pd.DataFrame]:
        """Return the measurements used, with missing values and
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
        self: Self,
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
        self.models = {}
