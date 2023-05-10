""" Contains classes and methods used to perform different methods of linear
regression

This module is used to perform different methods of linear regression on a
dataset (or a training subset), determine all coefficients and then calculate
a range of errors (using the testing subset if available).

    Classes:
        Calibration: Calibrates one set of measurements against another
"""

from copy import deepcopy as dc
import logging
from typing import Any, Literal

import bambi as bmb
import numpy as np
import pandas as pd
from sklearn import cross_decomposition as cd
from sklearn import ensemble as en
from sklearn import gaussian_process as gp
from sklearn import isotonic as iso
from sklearn import linear_model as lm
from sklearn import neural_network as nn
from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)


def cont_strat_folds(
        y_df: pd.DataFrame,
        target_var: str,
        splits: int = 5,
        strat_groups: int = 10,
        seed: int = 62
        ):
    """
    Creates stratified k-folds on continuous variable

    Parameters
    ----------
    y_df : pd.DataFrame
        Target data to stratify on
    target_var : str
        Target feature name
    splits : int, optional
        Number of folds to make
        Default is 5
    strat_groups : int, optional
        Number of groups to split data in to for stratification
        Default is 10
    seed : int, optional
        Random state to use, default is 62

    Returns
    -------
        y_df with added Fold column, specifying which test data fold
        variable corresponds to

    Based off of
    www.kaggle.com/code/tolgadincer/continuous-target-stratification/notebook
    """
    y_df['Fold'] = -1
    skf = StratifiedKFold(n_splits=splits, random_state=seed)
    y_df['Group'] = pd.cut(
            y_df[:, target_var],
            strat_groups,
            labels=False
            )
    target = y_df.loc[:, target_var]

    for fold_number, (_, v) in enumerate(skf.split(target, target)):
        y_df.loc[v, 'Fold'] = fold_number
    return y_df.drop('Group', axis=1)


class Calibrate:
    """

    ```

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(
            self,
            x_data: pd.DataFrame,
            y_data: pd.DataFrame,
            target: str,
            folds: int = 5,
            strat_groups: int = 10,
            seed: int = 62
                 ):
        """Initialises the calibration class

        This class is used to compare one set of measurements against another.
        It also has the capability to perform multivariate calibrations when
        secondary variables are provided.

        Parameters
        ----------
        x_data : pd.DataFrame
            Data to calibrate
        y_data : pd.DataFrame
            'True' data to calibrate against
        target : str
            Name of feature to calibrate, must be present in x and y
        folds : int, optional
            Number of fold, default is 5
        strat_groups : int, optional
            Number of groups to stratify against, default is 10
        seed : int, optional
            random state, default is 62
        """
        if target not in x_data.columns + y_data.columns:
            raise ValueError(
                    f"{target} does not exist in both columns,"
                    f" skipping comparison."
                             )
        join_index = x_data.join(
                y_data,
                how='inner',
                lsuffix='x',
                rsuffix='y'
                ).dropna().index
        self.x_data = x_data.loc[join_index, :]
        self.target = target
        self.y_data = cont_strat_folds(
                y_data.loc[join_index, :],
                target,
                folds,
                strat_groups,
                seed
                )
        self.y_data = self.y_data.drop(
                [
                    col for col in self.y_data.columns
                    if col not in ['Fold', target]
                    ]
                )
        self.models: dict[str, dict[int, Pipeline]] = dict()

    def _sklearn_regression_meta(
            self,
            reg: Any,
            name: str,
            min_coeffs: int = 1
            ):
        """
        Metaclass, formats data and uses sklearn classifier to
        fit x to y

        Parameters
        ----------
        reg : sklearn classifier
            Classifier to use
        name : str
            Name of classification technique to save pipeline to
        min_coeffs : int, optional
            Minimum number of coefficients for technique, default is 1
        """
        x_secondary_cols = self.x_data.drop(self.target, axis=1).columns
        products = [[np.nan, col] for col in x_secondary_cols]
        secondary_vals = pd.MultiIndex.from_product(products)
        for sec_vals in secondary_vals:
            vals = [self.target] + [v for v in sec_vals if v == v]
            if len(vals) < min_coeffs:
                continue
            model_name = f'{name} ({", ".join(vals)})'
            self.models[model_name] = dict()
            for fold in self.y_data.loc[:, 'Fold'].unique():
                y_data = self.y_data[
                        self.y_data.loc[:, 'Fold'] != fold
                        ].drop('Fold', axis=1)
                ss = StandardScaler()
                x_data = ss.fit_transform(
                        self.x_data.loc[y_data.index, vals]
                        )
                reg.fit(x_data, y_data)
                pipeline = Pipeline([
                    ("Selector", ColumnTransformer([
                            ("selector", "passthrough", vals)
                        ], remainder="drop")
                     ),
                    ("Standard Scaler", ss),
                    ("Regressor", reg)
                    ])
                self.models[model_name][fold] = dc(pipeline)

    def pymc_bayesian(
            self,
            family: Literal[
                "Gaussian",
                "Student T",
                ] = "Gaussian",
            **kwargs
            ):
        """Performs bayesian linear regression (either uni or multivariate)
        fitting x on y

        Performs bayesian linear regression, both univariate and multivariate,
        on X against y. More details can be found at:
        https://pymc.io/projects/examples/en/latest/generalized_linear_models/
        GLM-robust.html

        Parameters
        ----------
        family : Literal["Gaussian", "Student T"], optional
            Statistical distribution to fit measurements to. Options are:
                - Gaussian
                - Student T
            Default is "Gaussian"
        """
        # Define model families
        model_families = {
            "Gaussian": "gaussian",
            "Student T": "t",
            "Bernoulli": "bernoulli",
            "Beta": "beta",
            "Binomial": "binomial",
            "Gamma": "gamma",
            "Negative Binomial": "negativebinomial",
            "Poisson": "poisson",
            "Inverse Gaussian": "wald",
        }
        x_secondary_cols = self.x_data.drop(self.target, axis=1).columns
        products = [[np.nan, col] for col in x_secondary_cols]
        secondary_vals = pd.MultiIndex.from_product(products)
        for sec_vals in secondary_vals:
            vals = [self.target] + [v for v in sec_vals if v == v]
            model_name = f'Bayesian {family} ({", ".join(vals)})'
            self.models[model_name] = dict()
            for fold in self.y_data.loc[:, 'Fold'].unique():
                y_data = self.y_data[
                        self.y_data.loc[:, 'Fold'] != fold
                        ].drop('Fold', axis=1)
                ss = StandardScaler()
                x_data = ss.fit_transform(
                        self.x_data.loc[y_data.index, vals]
                        )
                x_data['y'] = y_data.loc[:, self.target]

                model = bmb.Model(
                        f"y ~ {' + '.join(vals)}",
                        x_data,
                        family=model_families[family]
                        )
                model.fit(
                        progressbar=False,
                        **kwargs
                        )
                pipeline = Pipeline([
                    ("Selector", ColumnTransformer([
                            ("selector", "passthrough", x_data.drop('y', vals))
                        ], remainder="drop")
                     ),
                    ("Standard Scaler", ss),
                    ("Regressor", model)
                    ])
                self.models[model_name][fold] = dc(pipeline)

    def linreg(self, name: str = "Linear Regression", **kwargs):
        """
        Fit x on y via linear regression

        Parameters
        ----------
        name : str, optional
            Name of classification technique, default is Linear Regression
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
        name : str, optional
            Name of classification technique, default is Ridge Regression
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
        name : str, optional
            Name of classification technique, default is Ridge Regression
            (Cross Validated)
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
        name : str, optional
            Name of classification technique, default is Lasso Regression
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
        name : str, optional
            Name of classification technique, default is Lasso Regression
            (Cross Validated)
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
        name : str, optional
            Name of classification technique, default is Multi-task Lasso
            Regression
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
        name : str, optional
            Name of classification technique, default is Multi-task Lasso
            Regression (Cross Validated)
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
        name : str, optional
            Name of classification technique, default is Multi-task Lasso
            Regression
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
        name : str, optional
            Name of classification technique, default is Elastic Net
            Regression (Cross Validated)
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
        name : str, optional
            Name of classification technique, default is Multi-task
            Elastic Net Regression
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
        name : str, optional
            Name of classification technique, default is Multi-task Elastic Net
            Regression (Cross Validated)
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
        name : str, optional
            Name of classification technique, default is Least Angle Regression
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
        name : str, optional
            Name of classification technique, default is Least Angle Regression
            (Lasso)
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
        name : str, optional
            Name of classification technique, default is Orthogonal Matching
            Pursuit
        """
        self._sklearn_regression_meta(
                lm.OrthogonalMatchingPursuit(**kwargs),
                name
                )

    def bayesian_ridge(self, name: str = "Bayesian Ridge Regression", **kwargs):
        """
        Fit x on y via bayesian ridge regression

        Parameters
        ----------
        name : str, optional
            Name of classification technique, default is Bayesian Ridge
            Regression
        """
        self._sklearn_regression_meta(
                lm.BayesianRidge(**kwargs),
                name
                )

    def bayesian_ard(self, name: str = "Bayesian Automatic Relevance Detection", **kwargs):
        """
        Fit x on y via bayesian automatic relevance detection

        Parameters
        ----------
        name : str, optional
            Name of classification technique, default is Bayesian Automatic
            Relevance Detection
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
        name : str, optional
            Name of classification technique, default is Tweedie Regression
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
        name : str, optional
            Name of classification technique, default is Stochastic Gradient
            Descent
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
        name : str, optional
            Name of classification technique, default is Passive Aggressive
            Regression
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
        name : str, optional
            Name of classification technique, default is RANSAC
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
        name : str, optional
            Name of classification technique, default is Theil-Sen Regression
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
        name : str, optional
            Name of classification technique, default is Huber Regression
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
        name : str, optional
            Name of classification technique, default is Quantile Regression
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
        name : str, optional
            Name of classification technique, default is Decision Tree
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
        name : str, optional
            Name of classification technique, default is Extra Tree
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
        name : str, optional
            Name of classification technique, default is Random Forest
        """
        self._sklearn_regression_meta(
                en.RandomForestRegressor(**kwargs),
                name
                )

    def extra_trees_enseble(self, name: str = "Extra Trees Ensemble", **kwargs):
        """
        Fit x on y using an ensemble of extra trees

        Parameters
        ----------
        name : str, optional
            Name of classification technique, default is Extra Trees Ensemble
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
        name : str, optional
            Name of classification technique, default is Gradient Boosting
            Regression
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
        name : str, optional
            Name of classification technique, default is Histogram-Based
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
        name : str, optional
            Name of classification technique, default is Multi-Layer Perceptron
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
        name : str, optional
            Name of classification technique, default is Support Vector
            Regression
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
        name : str, optional
            Name of classification technique, default is Linear Support Vector
            Regression
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
        name : str, optional
            Name of classification technique, default is Nu-Support Vector
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
        name : str, optional
            Name of classification technique, default is Gaussian Process
            Regression
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
        name : str, optional
            Name of classification technique, default is PLS Regression
        """
        self._sklearn_regression_meta(
                cd.PLSRegression(**kwargs),
                name
                )

    def isotonic(self, name: str = "Isotonic Regression", **kwargs):
        """
        Fit x on y using isotonic regression

        Parameters
        ----------
        name : str, optional
            Name of classification technique, default is Isotonic Regression
        """
        self._sklearn_regression_meta(
                iso.IsotonicRegression(**kwargs),
                name
                )

    def xgboost(self, name: str = "XGBoost Regression", **kwargs):
        """
        Fit x on y using xgboost regression

        Parameters
        ----------
        name : str, optional
            Name of classification technique, default is XGBoost Regression
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
        name : str, optional
            Name of classification technique, default is XGBoost Random Forest
            Regression
        """
        self._sklearn_regression_meta(
                xgb.XGBRFRegressor(**kwargs),
                name
                )

    def return_measurements(self) -> dict[str, pd.DataFrame]:
        return {
                'x': self.x_data,
                'y': self.y_data
                }
