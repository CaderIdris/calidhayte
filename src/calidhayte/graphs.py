import matplotlib.pyplot as plt
import pandas as pd


class Results:
    """Calculates errors between "true" and "predicted" measurements, plots
    graphs and returns all results

    Attributes:
        train (DataFrame): Training data

        test (DataFrame): Testing data

        coefficients (DataFrame): Calibration coefficients

        y_pred (dict): Calibrated x measurements

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
        train,
        test,
        coefficients,
        x_name=None,
        y_name=None,
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
        self._errors = defaultdict(lambda: defaultdict(list))
        self.y_pred = self._calibrate()
        self.combos = self._get_all_combos()
        self._plots = defaultdict(lambda: defaultdict(list))
        self.x_name = x_name
        self.y_name = y_name
    def linear_reg_plot(self, title=None):
        plot_name = "Linear Regression"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._plots[name]["Plot"]) == len(self._plots[name][method]):
                    self._plots[name]["Plot"].append(plot_name)
                if method != "x":
                    self._plots[name][method].append(None)
                    continue
                plt.style.use("Settings/style.mplstyle")
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

                max_value = max(max(true), max(pred))
                scatter_ax.set_xlim(0, max_value)
                scatter_ax.set_ylim(0, max_value)
                scatter_ax.set_xlabel(f"{self.x_name} ({method})")
                scatter_ax.set_ylabel(f"{self.y_name}")
                scatter_ax.scatter(pred, true, color="C0", alpha=0.75)
                number_of_coeffs = np.count_nonzero(
                    ~np.isnan(self.coefficients.loc[method].values)
                )
                if (
                    bool(re.search("Mean", name))
                    and not bool(re.search("Uncalibrated", name))
                    and number_of_coeffs == 4
                ):
                    scatter_ax.axline(
                        (0, self.coefficients.loc[method]["i.Intercept"]),
                        slope=self.coefficients.loc[method]["coeff.x"],
                        color="xkcd:vermillion",
                    )
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[method]["i.Intercept"]
                            + 2 * self.coefficients.loc[method]["sd.Intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[method]["coeff.x"]
                            + 2 * self.coefficients.loc[method]["sd.x"]
                        ),
                        color="xkcd:fresh green",
                    )
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[method]["i.Intercept"]
                            - 2 * self.coefficients.loc[method]["sd.Intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[method]["coeff.x"]
                            - 2 * self.coefficients.loc[method]["sd.x"]
                        ),
                        color="xkcd:fresh green",
                    )
                elif (
                    bool(re.search("Min", name))
                    and not bool(re.search("Uncalibrated", name))
                    and number_of_coeffs == 4
                ):
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[method]["i.Intercept"]
                            - 2 * self.coefficients.loc[method]["sd.Intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[method]["coeff.x"]
                            - 2 * self.coefficients.loc[method]["sd.x"]
                        ),
                        color="xkcd:vermillion",
                    )
                elif (
                    bool(re.search("Max", name))
                    and not bool(re.search("Uncalibrated", name))
                    and number_of_coeffs == 4
                ):
                    scatter_ax.axline(
                        (
                            0,
                            self.coefficients.loc[method]["i.Intercept"]
                            + 2 * self.coefficients.loc[method]["sd.Intercept"],
                        ),
                        slope=(
                            self.coefficients.loc[method]["coeff.x"]
                            + 2 * self.coefficients.loc[method]["sd.x"]
                        ),
                        color="xkcd:vermillion",
                    )
                elif (
                    not bool(re.search("Uncalibrated", name)) and number_of_coeffs == 2
                ):
                    scatter_ax.axline(
                        (0, int(self.coefficients.loc[method]["i.Intercept"])),
                        slope=self.coefficients.loc[method]["coeff.x"],
                        color="xkcd:vermillion",
                    )

                binwidth = 2.5
                xymax = max(np.max(np.abs(pred)), np.max(np.abs(true)))
                lim = (int(xymax / binwidth) + 1) * binwidth

                bins = np.arange(-lim, lim + binwidth, binwidth)
                histx_ax.hist(pred, bins=bins, color="C0")
                histy_ax.hist(true, bins=bins, orientation="horizontal", color="C0")
                if isinstance(title, str):
                    fig.suptitle(f"{title}\n{name} ({method})")

                self._plots[name][method].append(fig)

    def bland_altman_plot(self, title=None):
        plot_name = "Bland-Altman"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._plots[name]["Plot"]) == len(self._plots[name][method]):
                    self._plots[name]["Plot"].append(plot_name)
                plt.style.use("Settings/style.mplstyle")
                fig, ax = plt.subplots(figsize=(8, 8))
                x_data = np.mean(np.vstack((pred, true)).T, axis=1)
                y_data = np.array(pred) - np.array(true)
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
                    fig.suptitle(f"{title}\n{name} ({method})")

                self._plots[name][method].append(fig)

    def ecdf_plot(self, title=None):
        plot_name = "eCDF"
        for method, combo in self.combos.items():
            for name, pred, true in combo:
                if len(self._plots[name]["Plot"]) == len(self._plots[name][method]):
                    self._plots[name]["Plot"].append(plot_name)
                plt.style.use("Settings/style.mplstyle")
                fig, ax = plt.subplots(figsize=(8, 8))
                true_x, true_y = ecdf(true)
                pred_x, pred_y = ecdf(pred)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Measurement")
                ax.set_ylabel("Cumulative Total")
                ax.plot(true_x, true_y, linestyle="none", marker=".", label=self.y_name)
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
                    fig.suptitle(f"{title}\n{name} ({method})")
                self._plots[name][method].append(fig)

    def temp_time_series_plot(
        self, path, title=None
    ):  # This is not a good way to do this
        plt.style.use("Settings/style.mplstyle")
        x_vals = self.x_measurements["Values"]
        y_vals = self.y_measurements["Values"]
        try:
            dates_x = self.x_measurements["Datetime"]
            dates_y = self.y_measurements["Datetime"]
        except KeyError:
            dates_x = self.x_measurements.index
            dates_y = self.y_measurements.index
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(dates_y, y_vals, label=self.y_name)
        ax.plot(dates_x, x_vals, label=self.x_name)
        x_null = x_vals.isnull()
        y_null = y_vals.isnull()
        x_or_y_null = np.logical_or(x_null, y_null)
        first_datetime = dates_x[0]
        last_datetime = dates_x[-1]
        ax.legend()
        ax.set_xlim(first_datetime, last_datetime)
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Concentration")
        fig.savefig(f"{path}/Time Series.pgf")
        fig.savefig(f"{path}/Time Series.png")
        plt.close(fig)

    def save_plots(self, path):
        for key, item in self._plots.items():
            self._plots[key] = pd.DataFrame(data=dict(item))
            if "Plot" in self._plots[key].columns:
                self._plots[key] = self._plots[key].set_index("Plot")
            graph_types = self._plots[key].index.to_numpy()
            for graph_type in graph_types:
                graph_paths = dict()
                for vars, plot in self._plots[key].loc[graph_type].to_dict().items():
                    if plot is None:
                        continue
                    directory = Path(f"{path}/{key}/{vars}")
                    directory.mkdir(parents=True, exist_ok=True)
                    plot.savefig(f"{directory.as_posix()}/{graph_type}.pgf")
                    graph_paths[vars] = f"{directory.as_posix()}/{graph_type}.pgf"
                    plt.close(plot)
                    # key: Data set e.g uncalibrated full data
                    # graph_type: Type of graph e.g Linear Regression
                    # vars: Variables used e.g x + rh
                    # plot: The figure to be saved

    def save_results(self, path):
        for key, item in self._errors.items():
            self._errors[key] = pd.DataFrame(data=dict(item))
            if "Variable" in self._errors[key].columns:
                self._errors[key] = self._errors[key].set_index("Variable")
                vars_list = self._errors[key].columns.to_list()
                for vars in vars_list:
                    error_results = pd.DataFrame(self._errors[key][vars])
                    coefficients = pd.DataFrame(self.coefficients.loc[vars].T)
                    directory = Path(f"{path}/{key}/{vars}")
                    directory.mkdir(parents=True, exist_ok=True)
                    con = sql.connect(f"{directory.as_posix()}/Results.db")
                    error_results.to_sql(
                        name="Errors", con=con, if_exists="replace", index=True
                    )
                    coefficients.to_sql(
                        name="Coefficients", con=con, if_exists="replace", index=True
                    )
                    con.close()


def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y
