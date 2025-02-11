"""
manhattan_plot

==============

Manhattan plot for copy number data.
"""

from typing import Iterator, Self, Sequence

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

    CNV_LINE_COLOUR: str = "rgba(42,63,95,0.7)"

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

    def cnv_line_traces(
        self, copy_number: CopyNumber
    ) -> tuple[list[go.Scatter], list[go.Scatter]]:
        """Create CNV lines for manhattan plot.

        Parameters
        ----------
        copy_number : CopyNumber
            Copy number data. The class should have `manhattan_plot_preprocess()` method.

        Returns
        -------
        cnv_traces : list[go.Scatter]
            CNV lines for manhattan plot.
        non_cnv_traces : list[go.Scatter]
            Non-CNV lines for manhattan plot.
        """
        # CNV traces
        cnv_traces: list[go.Trace] = []
        cnv_traces_data: PolarsFrame = copy_number.manhattan_plot_preprocess()
        cnv_traces_df: pl.DataFrame = (
            cnv_traces_data.collect()
            if isinstance(cnv_traces_data, pl.LazyFrame)
            else cnv_traces_data
        )
        cnv_traces_df = (
            cnv_traces_df.join(
                other=self.position_converter, on="chr", how="left", coalesce=True
            )
            .with_columns(
                manhattan_start=pl.col("start") + pl.col("position_diff"),
                manhattan_end=pl.col("end") + pl.col("position_diff"),
            )
            .select(pl.col(["chr", "manhattan_start", "manhattan_end", "copy_number"]))
        )
        for curr_row in cnv_traces_df.iter_rows(named=True):
            curr_cnv_start: int = int(curr_row["manhattan_start"])
            curr_cnv_end: int = int(curr_row["manhattan_end"])
            curr_cnv_copy_number: float = float(curr_row["copy_number"])
            curr_row_trace_df = pl.DataFrame(
                {
                    "x": [curr_cnv_start, curr_cnv_end],
                    "y": [curr_cnv_copy_number] * 2,
                },
                schema=pl.Schema({"x": pl.Int64, "y": pl.Float32}),
                strict=True,
            )
            cnv_traces.append(
                go.Scattergl(
                    x=curr_row_trace_df["x"].to_list(),
                    y=curr_row_trace_df["y"].to_list(),
                    mode="lines",
                    name=f'cnv_{curr_row["chr"]}_{curr_cnv_start}_{curr_cnv_end}',
                    line=dict(color=self.CNV_LINE_COLOUR, width=4),
                    showlegend=False,
                )
            )

        # Non-CNV traces
        non_cnv_traces: list[go.Trace] = []

        def integers_to_intervals(integers: Sequence[int]) -> list[list[int]]:
            """
            Convert an ascending sequence of integers to a list of closed intervals.

            Parameters
            ----------
            integers : list[int]
                A sequence of ascending integers. For example, [1, 2, 3, 5, 6]

            Returns
            -------
            intervals : list[list[int]]
                A list of intervals. For example, [[1, 3], [5, 6]]
            """
            if len(integers) < 1:
                return []
            left: int = integers[0]
            right: int = left
            intervals: list[list[int]] = []

            i: int
            for i in integers[1:]:
                if i - right > 1:
                    intervals.append([left, right])
                    left = i
                right = i
            i = integers[-1]
            intervals.append([left, right])

            return intervals

        def normal_chr_copy_number(copy_numbers: Sequence[float]) -> int:
            """
            Convert a list of copy numbers to a single integer copy number for a entire normal chromosome.

            Parameters
            ----------
            copy_numbers : list[float]
                A list of copy numbers. For example, [1.0, 1.1, 1.3]

            Returns
            -------
            integer_copy_number : int
                The integer copy number. The result will only be 1 or 2.
            """
            supported_normal_chr_copy_numbers: list[int] = [0, 1, 2]
            max_chr_copy_number: int = int(max(supported_normal_chr_copy_numbers))
            min_chr_copy_number: int = int(min(supported_normal_chr_copy_numbers))
            copy_number_series: pl.Series = pl.Series(copy_numbers)
            copy_number_series_lower_quantile: float = copy_number_series.quantile(
                quantile=0.25, interpolation="lower"
            )
            copy_number_series_upper_quantile: float = copy_number_series.quantile(
                quantile=0.75, interpolation="higher"
            )
            filtered_copy_number_series: pl.Series = copy_number_series.filter(
                (copy_number_series >= copy_number_series_lower_quantile)
                & (copy_number_series <= copy_number_series_upper_quantile)
            )
            integer_copy_number: int = int(
                round(number=filtered_copy_number_series.mean(), ndigits=0)
            )
            integer_copy_number = int(min(integer_copy_number, max_chr_copy_number))
            integer_copy_number = int(max(integer_copy_number, min_chr_copy_number))
            return integer_copy_number

        for curr_trace in self.figure.data:
            if curr_trace.name.startswith("Chr") is False:
                continue
            curr_trace_chr: str = str(
                curr_trace.name.removeprefix("Chr")
            )  # Chr[number]. For example Chr1, Chr2, ..., Chr23, Chr24
            curr_trace_integer_copy_number: int = normal_chr_copy_number(curr_trace.y)
            curr_trace_x_min: int = int(curr_trace.x.min())
            curr_trace_x_max: int = int(curr_trace.x.max())

            # Search non-CNV intervals
            curr_trace_non_cnv_x_series: pl.Series = pl.int_range(
                start=curr_trace_x_min,
                end=curr_trace_x_max + 1,
                step=1,
                dtype=pl.Int64,
                eager=True,
            )
            for curr_cnv_trace in cnv_traces:
                curr_cnv_chr: str = curr_cnv_trace.name.removeprefix("cnv_")
                for origin_chr, replace_chr in copy_number._chromosome_rename.items():
                    if curr_cnv_chr.startswith(origin_chr + "_"):
                        curr_cnv_chr = curr_cnv_chr.replace(origin_chr, replace_chr, 1)
                if not (curr_cnv_chr.startswith(curr_trace_chr)):
                    continue
                curr_cnv_start = int(curr_cnv_trace.x[0])
                curr_cnv_end = int(curr_cnv_trace.x[1])
                curr_trace_non_cnv_x_series = curr_trace_non_cnv_x_series.filter(
                    (curr_trace_non_cnv_x_series < curr_cnv_start)
                    | (curr_trace_non_cnv_x_series > curr_cnv_end)
                )
            curr_trace_non_cnv_x_intervals = integers_to_intervals(
                curr_trace_non_cnv_x_series
            )

            # Draw non-CNV intervals
            for curr_trace_non_cnv_x_interval in curr_trace_non_cnv_x_intervals:
                curr_trace_non_cnv_x_interval_start: int = (
                    curr_trace_non_cnv_x_interval[0]
                )
                curr_trace_non_cnv_x_interval_end: int = curr_trace_non_cnv_x_interval[
                    1
                ]

                non_cnv_traces.append(
                    go.Scattergl(
                        x=[
                            curr_trace_non_cnv_x_interval_start,
                            curr_trace_non_cnv_x_interval_end,
                        ],
                        y=[
                            curr_trace_integer_copy_number,
                            curr_trace_integer_copy_number,
                        ],
                        mode="lines",
                        name=f"non_cnv_{curr_trace_chr}_{curr_trace_non_cnv_x_interval_start}_{curr_trace_non_cnv_x_interval_end}",
                        line=dict(color=self.CNV_LINE_COLOUR, width=4),
                        showlegend=False,
                    )
                )

        return cnv_traces, non_cnv_traces

    def add_cnv_line(self, copy_number: CopyNumber) -> Self:
        """Add CNV lines to a manhattan plot."""
        cnv_traces, non_cnv_traces = self.cnv_line_traces(copy_number=copy_number)
        self.figure = self.figure.add_traces(cnv_traces)
        self.figure = self.figure.add_traces(non_cnv_traces)
        return self
