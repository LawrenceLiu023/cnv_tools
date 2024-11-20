"""
manhattan_plot

==============

Manhattan plot for copy number data.
"""

from typing import Self

import plotly.graph_objects as go
import polars as pl

from cnv_tools._typing import PolarsFrame
from cnv_tools.copy_number import CopyNumber


class ManhattanPlot:
    """
    Manhattan plot for copy number data.

    Attributes
    ----------
    figure : go.Figure
        Plotly figure.
    position_converter : pl.DataFrame
        DataFrame with chromosome and position conversion information.

    Methods
    -------
    from_copy_number(copy_number: CopyNumber) -> ManhattanPlot
        Create a Manhattan plot from copy number data.
    cnv_line_traces(copy_number: CopyNumber) -> list[go.Scatter]
        Create CNV lines for manhattan plot.
    add_cnv_line(copy_number: CopyNumber) -> ManhattanPlot:
        Add CNV lines to a manhattan plot.
    """

    def __init__(self, figure: go.Figure, position_converter: PolarsFrame) -> None:
        self.figure: go.Figure = figure
        self.position_converter: pl.DataFrame = (
            position_converter.collect()
            if isinstance(position_converter, pl.LazyFrame)
            else position_converter
        )

    @classmethod
    def from_copy_number(cls, copy_number: CopyNumber) -> Self:
        """Create a Manhattan plot from copy number data."""
        manhattan_plot: ManhattanPlot = copy_number.manhattan_plot()
        return manhattan_plot

    def cnv_line_traces(self, copy_number: CopyNumber) -> list[go.Scatter]:
        """Create CNV lines for manhattan plot."""
        cnv_traces: list[go.Trace] = []
        cnv_traces_data: PolarsFrame = copy_number.manhattan_plot_preprocess()
        cnv_traces_data = cnv_traces_data.join(
            other=self.position_converter, on="chr", how="left", coalesce=True
        )
        cnv_traces_data = cnv_traces_data.with_columns(
            manhattan_start=pl.col("start") + pl.col("position_diff"),
            manhattan_end=pl.col("end") + pl.col("position_diff"),
        ).select(pl.col(["chr", "manhattan_start", "manhattan_end", "copy_number"]))
        cnv_traces_df: pl.DataFrame = (
            cnv_traces_data.collect()
            if isinstance(cnv_traces_data, pl.LazyFrame)
            else cnv_traces_data
        )
        for curr_row in cnv_traces_df.iter_rows(named=True):
            curr_row_trace_df = pl.DataFrame(
                {
                    "x": [curr_row["manhattan_start"], curr_row["manhattan_end"]],
                    "y": [curr_row["copy_number"]] * 2,
                },
                schema=pl.Schema({"x": pl.Int64, "y": pl.Float32}),
                strict=True,
            )
            cnv_traces.append(
                go.Scattergl(
                    x=curr_row_trace_df["x"].to_list(),
                    y=curr_row_trace_df["y"].to_list(),
                    mode="lines",
                    name=curr_row["chr"],
                    line=dict(color="rgba(42,63,95,0.7)", width=4),
                    showlegend=False,
                )
            )
        return cnv_traces

    def add_cnv_line(self, copy_number: CopyNumber) -> Self:
        """Add CNV lines to a manhattan plot."""
        self.figure = self.figure.add_traces(
            self.cnv_line_traces(copy_number=copy_number)
        )
        return self
