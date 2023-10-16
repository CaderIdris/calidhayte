from collections.abc import Iterable
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from matplotlib import get_backend
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import shap
from sklearn.pipeline import Pipeline


class Graphs:
    """
    Calculates errors between "true" and "predicted" measurements, plots
    graphs and returns all results
    """

    def __init__(
        self,
        x: pd.DataFrame,
        x_name: str,
        y: pd.DataFrame,
        y_name: str,
        target: str,
        models: dict[str, dict[str, dict[str, dict[int, Pipeline]]]],
        style: str = 'bmh',
        backend: str = str(get_backend())
    ):
        """
        """
        self.x: pd.DataFrame = x
        """
        Independent variable(s) that are calibrated against `y`,
        the independent variable. Index should match `y`.
        """
        self.y: pd.DataFrame = y
        """
        Dependent variable used to calibrate the independent variables `x`.
        Index should match `x`.
        """
        self.x_name: str = x_name
        """
        Label for `x` measurements
        """
        self.y_name: str = y_name
        """
        Label for `y` measurements
        """
        self.target = target
        """
        Measurand in `y` to calibrate against
        """
        self.models: dict[
            str, dict[  # Scaling Method
                str, dict[  # Variables used
                    str, dict[  # Fold
                        int, Pipeline]]]] = models
        """
        The precalibrated models. They are stored in a nested structure as
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
        self.plots: dict[str,  # Technique
                         dict[str,  # Scaling Method
                              dict[str,  # Variables used
                                   dict[str,  # Plot Name
                                        matplotlib.figure.Figure]]]] = dict()
        """
        The plotted data, stored in a similar structure to `models`
        1. Primary Key, name of the technique (e.g Lasso Regression).
        2. Scaling technique (e.g Yeo-Johnson Transform).
        3. Combination of variables used or `target` if calibration is
        univariate (e.g "`target` + a + b).
        4. Name of the plot (e.g. 'Bland-Altman')

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
                    [*] --> pn
                    state "Plot Name" as pn {
                     [*] : Name of the plot
                     [*] : (e.g Bland-Altman)
                    }
                  }
                }
              }
        ```

        """
        self.style: Union[str, Path] = style
        """
        Name of in-built matplotlib style or path to stylesheet
        """
        self.backend = backend
        """
        Matplotlib backend to use
        """

    def plot_meta(
        self,
        plot_func: Callable[
            ...,
            matplotlib.figure.Figure
        ],
        name: str,
        **kwargs
    ):
        """
        Iterates over data and creates plots using function specified in
        `plot_func`

        Should not be accessed directly, should instead be called by
        another method

        Parameters
        ----------
        plot_func : Callable
            Function that returns matplotlib figure
        name : str
            Name to give plot, used as key in `plots` dict
        **kwargs
            Additional arguments passed to `plot_func`
        """
        if not self.x.sort_index().index.to_series().eq(
            self.y.sort_index().index.to_series()
        ).all():
            raise ValueError(
                'Index of x and y do not match. Output of Calibrate class '
                'in calidhayte should have matching indexes'
            )
        for technique, scaling_methods in self.models.items():
            if self.plots.get(technique) is None:
                self.plots[technique] = dict()
            for scaling_method, var_combos in scaling_methods.items():
                if self.plots[technique].get(scaling_method) is None:
                    self.plots[technique][scaling_method] = dict()
                for vars, folds in var_combos.items():
                    if self.plots[technique][scaling_method].get(vars) is None:
                        self.plots[technique][scaling_method][vars] = dict()
                    pred = pd.Series()
                    for fold, model in folds.items():
                        x_data = self.x.loc[
                                self.y[self.y.loc[:, 'Fold'] == fold].index,
                                :
                                ]
                        pred = pd.concat(
                                [
                                    pred,
                                    pd.Series(
                                        index=x_data.index,
                                        data=model.predict(x_data)
                                        )
                                ]
                            )
                    x = pred
                    y = self.y.loc[:, self.target].reindex(x.index)
                    fig = plot_func(
                            x=x,
                            y=y,
                            x_name=self.x_name,
                            y_name=self.y_name,
                            **kwargs
                            )
                    self.plots[technique][scaling_method][vars][name] = fig

    def bland_altman_plot(self, title=None):
        with plt.rc_context({'backend': self.backend}), \
                plt.style.context(self.style):
            self.plot_meta(bland_altman_plot, 'Bland-Altman', title=title)

    def ecdf_plot(self, title=None):
        with plt.rc_context({'backend': self.backend}), \
                plt.style.context(self.style):
            self.plot_meta(ecdf_plot, 'eCDF', title=title)

    def lin_reg_plot(self, title=None):
        with plt.rc_context({'backend': self.backend}), \
                plt.style.context(self.style):
            self.plot_meta(lin_reg_plot, 'Linear Regression', title=title)

    def save_plots(
        self,
        path: str,
        filetype: Union[
           Literal['png', 'pgf', 'pdf'],
           Iterable[Literal['png', 'pgf', 'pdf']]
            ] = 'png'
    ):
        for technique, scaling_methods in self.plots.items():
            for scaling_method, var_combos in scaling_methods.items():
                for vars, figures in var_combos.items():
                    for plot_type, fig in figures.items():
                        plot_path = Path(
                                f'{path}/{technique}/{plot_type}'
                                )
                        plot_path.mkdir(parents=True, exist_ok=True)
                        if isinstance(filetype, str):
                            fig.savefig(
                                plot_path /
                                f'{scaling_method} {vars}.{filetype}'
                            )
                        elif isinstance(filetype, Iterable):
                            for ftype in filetype:
                                fig.savefig(
                                    plot_path /
                                    f'{scaling_method} {vars}.{ftype}'
                                )
                        plt.close(fig)


def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y


def lin_reg_plot(
        x: pd.Series,
        y: pd.Series,
        x_name: str,
        y_name: str,
        title: Optional[str] = None
        ):
    """
    """
    fig = plt.figure(figsize=(4, 4), dpi=200)
    fig_gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(7, 2),
        height_ratios=(2, 7),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.0,
        hspace=0.0,
    )

    scatter_ax = fig.add_subplot(fig_gs[1, 0])
    histx_ax = fig.add_subplot(fig_gs[0, 0], sharex=scatter_ax)
    histx_ax.axis("off")
    histy_ax = fig.add_subplot(fig_gs[1, 1], sharey=scatter_ax)
    histy_ax.axis("off")

    max_value = max((y.max(), x.max()))
    min_value = min((y.min(), x.min()))
    scatter_ax.set_xlim(min_value - 3, max_value + 3)
    scatter_ax.set_ylim(min_value - 3, max_value + 3)
    scatter_ax.set_xlabel(x_name)
    scatter_ax.set_ylabel(y_name)
    scatter_ax.scatter(x, y, color="C0", marker='.', alpha=0.75)

    binwidth = 7.5
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = list(np.arange(-lim, lim + binwidth, binwidth))
    histx_ax.hist(x, bins=bins, color="C0")
    histy_ax.hist(y, bins=bins, orientation="horizontal", color="C0")
    if isinstance(title, str):
        fig.suptitle(title)
    return fig


def bland_altman_plot(
        x: pd.DataFrame,
        y: pd.Series,
        title: Optional[str] = None,
        **kwargs
        ):
    """
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    x_data = np.mean(np.vstack((x, y)).T, axis=1)
    y_data = np.array(x) - np.array(y)
    y_mean = np.mean(y_data)
    y_sd = 1.96 * np.std(y_data)
    max_diff_from_mean = max(
        (y_data - y_mean).min(), (y_data - y_mean).max(), key=abs
    )
    text_adjust = (12 * max_diff_from_mean) / 300
    ax.set_ylim(y_mean - max_diff_from_mean, y_mean + max_diff_from_mean)
    ax.set_xlabel("Average of Measured and Reference")
    ax.set_ylabel("Difference Between Measured and Reference")
    ax.scatter(x_data, y_data, alpha=0.75)
    ax.axline((0, y_mean), (1, y_mean), color="xkcd:vermillion")
    ax.text(
        max(x_data),
        y_mean + text_adjust,
        f"Mean: {y_mean:.2f}",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax.axline(
        (0, y_mean + y_sd), (1, y_mean + y_sd), color="xkcd:fresh green"
    )
    ax.text(
        max(x_data),
        y_mean + y_sd + text_adjust,
        f"1.96$\\sigma$: {y_mean + y_sd:.2f}",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax.axline(
        (0, y_mean - y_sd), (1, y_mean - y_sd), color="xkcd:fresh green"
    )
    ax.text(
        max(x_data),
        y_mean - y_sd + text_adjust,
        f"1.96$\\sigma$: -{y_sd:.2f}",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    if isinstance(title, str):
        fig.suptitle(title)
    return fig


def ecdf_plot(
        x: pd.DataFrame,
        y: pd.Series,
        x_name: str,
        y_name: str,
        title: Optional[str] = None
        ):
    """
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    true_x, true_y = ecdf(y)
    pred_x, pred_y = ecdf(x)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Measurement")
    ax.set_ylabel("Cumulative Total")
    ax.plot(true_x, true_y, linestyle="none", marker=".", label=y_name)
    ax.plot(
        pred_x,
        pred_y,
        linestyle="none",
        marker=".",
        alpha=0.8,
        label=x_name,
    )
    ax.legend()
    if isinstance(title, str):
        fig.suptitle(title)
    return fig
