from pathlib import Path
import re
from typing import Literal, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calibrate import Calibrate

mpl.use("pgf")  # Used to make pgf files for latex
plt.rcParams.update({"figure.max_open_warning": 0})


class Results:
    """Calculates errors between "true" and "predicted" measurements, plots
    graphs and returns all results

    Attributes:
        train (DataFrame): Training data

        test (DataFrame): Testing data

        coefficients (DataFrame): Calibration coefficients

        y_dset['x'].loc[:, var] (dict): Calibrated x measurements

        _plots (dict): All result plots made

        x_name (str): Name of x device

        y_name (str): Name of y device

    Methods:
        bland_altman_plot: Plots a bland altman graph for all variable
        combinations for all specified datasets using predicted (calibrated x)
        and true (y) data

        linear_reg_plot: Plots a linear regression graph for calibrations that
        only have an x coefficients for all specified datasets using predited
        (calibrated x) and true (y) data

        ecdf_plot: Plots an eCDF graph for all variable combinations for all
        specified dataset using predicted (calibrated x) and true (y) data

        temp_time_series_plot: Temporary way to plot time series, not great

        save_results: Saves errors and coefficients for specific variable and
        dataset to local sqlite3 file

        save_plots: Saves all plots in pgf format
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        coefficients: pd.DataFrame,
        datasets_to_use: list[
            Literal[
                "Calibrated Train",
                "Calibrated Test",
                "Calibrated Full",
                "Uncalibrated Train",
                "Uncalibrated Test",
                "Uncalibrated Full",
                "Minimum",
                "Maximum"
                ]
            ] = [
            "Calibrated Test",
            "Uncalibrated Test"
        ],
        style: str = "bmh",
        x_name: Optional[str] = None,
        y_name: Optional[str] = None
    ):
        """Initialise the class

        Keyword Arguments:
            train (DataFrame): Training data

            test (DataFrame): Testing data

            coefficients (DataFrame): Calibration coefficients

            comparison_name (String): Name of the comparison
        """
        self.train = train
        self.test = test
        self.coefficients = coefficients
        self._cal = Calibrate(
                self.train,
                self.test,
                self.coefficients
                )
        self.y_subsets = self._cal.return_measurements()
        self.y_full = self._cal.join_measurements()
        self._datasets = self._prepare_datasets(datasets_to_use)

        self.style = style
        self.x_name = x_name
        self.y_name = y_name

        self._plots: dict[str, dict[str, dict[str, mpl.figure]]] = dict()

    def _prepare_datasets(
            self,
            datasets_to_use: list[
                Literal[
                    "Calibrated Train",
                    "Calibrated Test",
                    "Calibrated Full",
                    "Uncalibrated Train",
                    "Uncalibrated Test",
                    "Uncalibrated Full",
                    "Minimum",
                    "Maximum"
                    ]
                ]
            ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Prepare the datasets to be analysed

        Parameters
        ----------
        datasets_to_use : list[str]
            List containing datasets to use
            Datasets are as follows:
                - Calibrated Train
                    predicted y measurements from training set using
                    coefficients
                - Calibrated Test
                    predicted y measurements from testing set using
                    coefficients
                - Calibrated Full
                    predicted y measurements from both sets using
                    coefficients
                - Uncalibrated Train
                    Base x measurements from training set
                - Uncalibrated Test
                    Base x measurements from testing set
                - Uncalibrated Full
                    Base x measurements from testing set
                - Minimum
                    Mean bayesian coefficients - 2 times standard deviation
                - Maximum
                    Mean bayesian coefficients + 2 times standard deviation
        Returns
        -------
        dict[str, dict[str, pd.DataFrame]]
            Contains all subsets of the data to be analysed
        """
        uncalibrated_datasets = {
                "Uncalibrated Train": {
                    "x": self.train.loc[:, ['x']],
                    "y": self.train.loc[:, ['y']]
                    },
                "Uncalibrated Test": {
                    "x": self.test.loc[:, ['x']],
                    "y": self.test.loc[:, ['y']]
                    },
                "Uncalibrated Full": {
                    "x": self.y_full['Uncalibrated'].loc[:, ['x']],
                    "y": self.y_full['Uncalibrated'].loc[:, ['y']]
                    }
                }
        pymc_bool = all(
                [
                    any(["Mean." in key for key in self.y_subsets.keys()]),
                    any(["Minimum." in key for key in self.y_subsets.keys()]),
                    any(["Maximum." in key for key in self.y_subsets.keys()])
                    ]
                )
        if pymc_bool:
            cal_datasets = {
                    "Calibrated Train (Mean)": {
                        "x": self.y_subsets['Mean.Train'],
                        "y": self.train.loc[:, ['y']]
                        },
                    "Calibrated Test (Mean)": {
                        "x": self.y_subsets['Mean.Test'],
                        "y": self.test.loc[:, ['y']]
                        },
                    "Calibrated Full (Mean)": {
                        "x": self.y_full['Mean.Calibrated'],
                        "y": self.y_full['Uncalibrated'].loc[:, ['y']]
                        },
                    "Calibrated Train (Minimum)": {
                        "x": self.y_subsets['Minimum.Train'],
                        "y": self.train.loc[:, ['y']]
                        },
                    "Calibrated Test (Minimum)": {
                        "x": self.y_subsets['Minimum.Test'],
                        "y": self.test.loc[:, ['y']]
                        },
                    "Calibrated Full (Minimum)": {
                        "x": self.y_full['Minimum.Calibrated'],
                        "y": self.y_full['Uncalibrated'].loc[:, ['y']]
                        },
                    "Calibrated Train (Maximum)": {
                        "x": self.y_subsets['Maximum.Train'],
                        "y": self.train.loc[:, ['y']]
                        },
                    "Calibrated Test (Maximum)": {
                        "x": self.y_subsets['Maximum.Test'],
                        "y": self.test.loc[:, ['y']]
                        },
                    "Calibrated Full (Maximum)": {
                        "x": self.y_full['Maximum.Calibrated'],
                        "y": self.y_full['Uncalibrated'].loc[:, ['y']]
                        },
                    }
        else:
            cal_datasets = {
                    "Calibrated Train": {
                        "x": self.y_subsets['Train'],
                        "y": self.train.loc[:, ['y']]
                        },
                    "Calibrated Test": {
                        "x": self.y_subsets['Test'],
                        "y": self.test.loc[:, ['y']]
                        },
                    "Calibrated Full": {
                        "x": self.y_full['Calibrated'],
                        "y": self.y_full['Uncalibrated'].loc[:, ['y']]
                        }
                    }
        datasets = uncalibrated_datasets | cal_datasets
        uncal_sets = filter(
                lambda x: bool(re.search(r'^Uncalibrated ', x)),
                datasets_to_use
                )
        cal_sets = filter(
                lambda x: bool(re.search(r'^Calibrated ', x)),
                datasets_to_use
                )
        selected_datasets = dict()
        for uncal_key in uncal_sets:
            selected_datasets[str(uncal_key)] = datasets[uncal_key]
        if pymc_bool:
            min_max_sets = list(

                    filter(
                        lambda x: bool(re.search(r'^Minimum|^Maximum', x)),
                        datasets_to_use
                    )
                )
            for cal_key in cal_sets:
                for pymc_subset in ['Mean'] + min_max_sets:
                    selected_datasets[
                            f'{cal_key} ({pymc_subset})'
                            ] = datasets[f'{cal_key} ({pymc_subset})']
        else:
            for cal_key in cal_sets:
                selected_datasets[str(cal_key)] = datasets[cal_key]

        return selected_datasets

    def linear_reg_plot(self, title: Optional[str] = None):
        for dset_key, dset in self._datasets.items():
            if dset_key not in self._plots.keys():
                self._plots[dset_key] = dict()
            for var in dset['x'].columns:
                number_of_coeffs = self.coefficients.loc[var, :].notna().sum()
                pymc_bool = self.coefficients.filter(
                        regex=r'^sd\.', axis=1
                        ).shape[1]
                if (
                        (pymc_bool and number_of_coeffs != 4)
                        or
                        (not pymc_bool and number_of_coeffs != 2)
                        ):
                    continue
                if var not in self._plots[dset_key].keys():
                    self._plots[dset_key][var] = dict()
                plt.style.use(self.style)
                fig = plt.figure(figsize=(8, 8))
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

                max_value = max(
                        max(dset['y'].loc[:, 'y']),
                        max(dset['x'].loc[:, var])
                        )
                scatter_ax.set_xlim(0, max_value)
                scatter_ax.set_ylim(0, max_value)
                scatter_ax.set_xlabel(f"{self.x_name} (x)")
                scatter_ax.set_ylabel(f"{self.y_name}")
                scatter_ax.scatter(
                        dset['x'].loc[:, var],
                        dset['y'].loc[:, 'y'],
                        alpha=0.75)
                if bool(re.search(r' \(Mean\)$', dset_key)):
                    scatter_ax.axline(
                        (0, self.coefficients.loc[var]["i.intercept"]),
                        slope=self.coefficients.loc[var]["coeff.x"]
                    )
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[var]["i.intercept"]
                            + 2 * self.coefficients.loc[var]["sd.Intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[var]["coeff.x"]
                            + 2 * self.coefficients.loc[var]["sd.x"]
                        ),
                        color="xkcd:fresh green",
                    )
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[var]["i.intercept"]
                            - 2 * self.coefficients.loc[var]["sd.intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[var]["coeff.x"]
                            - 2 * self.coefficients.loc[var]["sd.x"]
                        ),
                        color="xkcd:fresh green",
                    )
                elif bool(re.search(r' \(Minimum\)$', dset_key)):
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[var]["i.intercept"]
                            - 2 * self.coefficients.loc[var]["sd.intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[var]["coeff.x"]
                            - 2 * self.coefficients.loc[var]["sd.x"]
                        ),
                    )
                elif bool(re.search(r' \(Maximum\)$', dset_key)):
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[var]["i.intercept"]
                            + 2 * self.coefficients.loc[var]["sd.intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[var]["coeff.x"]
                            + 2 * self.coefficients.loc[var]["sd.x"]
                        ),
                    )
                elif not bool(re.search(r'^Uncalibrated', dset_key)):
                    scatter_ax.axline(
                        (0, int(self.coefficients.loc[var]["i.intercept"])),
                        slope=self.coefficients.loc[var]["coeff.x"],
                    )

                binwidth = 2.5
                xymax = max(
                        np.max(np.abs(dset['x'].loc[:, var])),
                        np.max(np.abs(dset['y'].loc[:, 'y']))
                        )
                lim = (int(xymax / binwidth) + 1) * binwidth

                bins = np.arange(-lim, lim + binwidth, binwidth)
                histx_ax.hist(
                        dset['x'].loc[:, var],
                        bins=bins,
                        color="C0"
                        )
                histy_ax.hist(
                        dset['y'].loc[:, 'y'],
                        bins=bins,
                        orientation="horizontal",
                        color="C0"
                        )
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{dset_key} ({var})")

                self._plots[dset_key][var]['Linear Regression'] = fig

    def bland_altman_plot(self, title: Optional[str] = None):
        for dset_key, dset in self._datasets.items():
            if dset_key not in self._plots.keys():
                self._plots[dset_key] = dict()
            for var in dset['x'].columns:
                if var not in self._plots[dset_key].keys():
                    self._plots[dset_key][var] = dict()
                plt.style.use(self.style)
                fig, ax = plt.subplots(figsize=(8, 8))
                x_data = dset['x'].loc[:, var].join(dset['y'].loc['y']).mean()
                y_data = dset['x'].loc[:, var] - dset['y'].loc[:, 'y']
                y_mean = np.mean(y_data)
                y_sd = 1.96 * np.std(y_data)

                max_diff_from_mean = max(
                    (y_data - y_mean).min(), (y_data - y_mean).max(), key=abs
                )

                text_adjust = (12 * max_diff_from_mean) / 300
                ax.set_ylim(
                        y_mean - max_diff_from_mean,
                        y_mean + max_diff_from_mean
                        )
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
                    (0, y_mean + y_sd),
                    (1, y_mean + y_sd),
                    color="xkcd:fresh green"
                )
                ax.text(
                    max(x_data),
                    y_mean + y_sd + text_adjust,
                    f"1.96$\\sigma$: {y_mean + y_sd:.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )
                ax.axline(
                    (0, y_mean - y_sd),
                    (1, y_mean - y_sd),
                    color="xkcd:fresh green"
                )
                ax.text(
                    max(x_data),
                    y_mean - y_sd + text_adjust,
                    f"1.96$\\sigma$: -{y_sd:.2f}",
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{dset_key} ({var})")

                self._plots[dset_key][var]['Bland Altman'] = fig

    def ecdf_plot(self, title=None):
        for dset_key, dset in self._datasets.items():
            if dset_key not in self._plots.keys():
                self._plots[dset_key] = dict()
            for var in dset['x'].columns:
                if var not in self._plots[dset_key].keys():
                    self._plots[dset_key][var] = dict()
                plt.style.use(self.style)
                fig, ax = plt.subplots(figsize=(8, 8))
                true_x, true_y = ecdf(dset['y'].loc[:, 'y'])
                pred_x, pred_y = ecdf(dset['x'].loc[:, var])
                ax.set_ylim(0, 1)
                ax.set_xlabel("Measurement")
                ax.set_ylabel("Cumulative Total")
                ax.plot(
                        true_x,
                        true_y,
                        linestyle="none",
                        marker=".",
                        label=self.y_name
                        )
                ax.plot(
                    pred_x,
                    pred_y,
                    linestyle="none",
                    marker=".",
                    alpha=0.8,
                    label=self.x_name,
                )
                ax.legend()
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{dset_key} ({var})")
                self._plots[dset_key][var]['eCDF'] = fig

    def time_series_plot(self, title: Optional[str] = None):
        for dset_key, dset in self._datasets.items():
            if not bool(re.search(r' Full', dset_key)):
                continue
            if dset_key not in self._plots.keys():
                self._plots[dset_key] = dict()
            for var in dset['x'].columns:
                if var not in self._plots[dset_key].keys():
                    self._plots[dset_key][var] = dict()
                plt.style.use(self.style)
                x_vals = dset['x'].loc[:, var]
                y_vals = dset['y'].loc[:, 'y']
                dates_x = x_vals.index
                dates_y = y_vals.index
                fig, ax = plt.subplots(figsize=(16, 8))
                ax.plot(dates_y, y_vals, label=self.y_name)
                ax.plot(dates_x, x_vals, label=self.x_name)
                x_null = x_vals.isnull()
                y_null = y_vals.isnull()
                combined_dates = (x_null and y_null)
                first_datetime = x_vals.loc[combined_dates].index[0]
                last_datetime = x_vals.loc[combined_dates].index[-1]
                ax.legend()
                ax.set_xlim(first_datetime, last_datetime)
                ax.set_xlabel("Datetime")
                ax.set_ylabel("Concentration")
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{dset_key} ({var})")
                self._plots[dset_key][var]['Time Series'] = fig

    def save_plots(self, path: str, format: str = "pgf"):
        for dset_key, dset_plots in self._plots.items():
            for var, plots in dset_plots.items():
                for name, plot in plots.items():
                    directory = Path(f"{path}/{dset_key}/{var}")
                    directory.mkdir(parents=True, exist_ok=True)
                    plot.savefig(f"{directory.as_posix()}/{name}.{format}")
                    plt.close(plot)


def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y
