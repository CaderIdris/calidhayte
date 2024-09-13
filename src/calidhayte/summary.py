import pathlib

from matplotlib import get_backend
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd


class Summary:
    """ """

    def __init__(
        self,
        results: pd.DataFrame,
        cols: list[str],
        style: str = "bmh",
        backend: str = str(get_backend()),
    ):
        """ """
        self.results = results
        self.plots: dict[str, dict[str, matplotlib.figure.Figure]] = dict()
        self.cols: list[str] = cols
        self.style = style
        self.backend = backend

    def boxplots(self):
        """ """
        self.plots["Box Plots"] = dict()
        for label in self.results.index.names[:-1]:
            for col in self.cols:
                with plt.rc_context(
                    {"backend": self.backend, "figure.dpi": 200}
                ), plt.style.context(self.style):
                    plot = self.results.loc[:, [col]].boxplot(
                        by=label,
                        figsize=(
                            len(self.cols),
                            2 * round(len(self.cols) / 2),
                        ),
                        rot=90,
                        fontsize=8,
                        sym=".",
                    )
                    plot.title.set_size(8)
                    plt.tight_layout()
                    self.plots["Box Plots"][f"{label} {col}"] = plot
                    plt.close()

    def histograms(self):
        """ """
        self.plots["Histograms"] = dict()
        for col in self.cols:
            with plt.rc_context(
                {"backend": self.backend, "figure.dpi": 200}
            ), plt.style.context(self.style):
                plot = self.results.loc[:, col].plot.hist(
                    bins=30, figsize=(8, 4)
                )
                plot.set_xlabel(col)
                plot.title.set_size(8)
                plt.tight_layout()
                self.plots["Histograms"][col] = plot
                plt.close()

    def save_plots(self, path, filetype: str = "png"):
        """ """
        for plot_type, plots in self.plots.items():
            for variable, ax in plots.items():
                plot_path = pathlib.Path(f"{path}/Summary")
                fig = ax.figure
                plot_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(plot_path / f"{plot_type} {variable}.{filetype}")
                plt.close(fig)
