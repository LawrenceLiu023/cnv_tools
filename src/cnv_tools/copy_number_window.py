"""
copy_number_window

==================

Copy number data of genome windows.
"""

from typing import Literal, Self, Sequence

import dash_bio
import numpy as np
import plotly.graph_objects as go
import polars as pl
from sklearn import metrics

from cnv_tools._typing import PolarsFrame
from cnv_tools.copy_number import CopyNumber
from cnv_tools.manhattan_plot import ManhattanPlot


class CopyNumberWindow(CopyNumber):
    """
    Copy number information of genome windows.

    Attributes
    ----------
    data : PolarsFrame
        Copy number data of genome windows. A polars DataFrame / LazyFrame containing the following columns:

        - "chr": polars.String
        - "start": polars.Int64
        - "end": polars.Int64
        - "copy_number": polars.Float32
        - "integer_copy_number": polars.Int32

        The "chr" column are like "1", "2", ..., "X", "Y". The rows are sorted by the "chr" and "start" columns. The "chr" column order are like "1", "2", ..., "22", "X", "Y". Other columns will not be modified.

    Methods
    -------
    region_consistency_check(copy_number_x: CopyNumberWindow, copy_number_y: CopyNumberWindow) -> bool
        Check if the regions (genome windows or chromosome names) of two copy number data are consistent.
    accuracy_score(copy_number_true: CopyNumberWindow, copy_number_pred: CopyNumberWindow) -> float
        Calculate the accuracy score of two copy number data.
    recall_score(copy_number_true: CopyNumberWindow, copy_number_pred: CopyNumberWindow, average: Literal["micro", "samples", "weighted", "macro"] | None) -> float
        Calculate the recall score of two copy number data.
    precision_score(copy_number_true: CopyNumberWindow, copy_number_pred: CopyNumberWindow, average: Literal["micro", "samples", "weighted", "macro"] | None) -> float
        Calculate the precision score of two copy number data.
    difference_std(copy_number_true: CopyNumberWindow, copy_number_pred: CopyNumberWindow) -> float
        Calculate the standard deviation of the difference between two copy number data.
    manhattan_plot_preprocess() -> PolarsFrame
        Preprocess the data for manhattan plot.
    manhattan_plot() -> ManhattanPlot
        Draw manhattan plot of copy number.
    """

    def __init__(
        self,
        copy_number_data: PolarsFrame,
        chr_col: str = "chr",
        start_col: str = "start",
        end_col: str = "end",
        copy_number_col: str = "copy_number",
    ) -> None:
        """
        Create a `CopyNumberWindow`.

        Parameters
        ----------
        copy_number_data : PolarsFrame
            A polars DataFrame / LazyFrame containing the following information:

            - chromosome
            - start position
            - end position
            - copy number

        chr_col : str, default "chr"
            The column name of the chromosome column. Supported chromosome formats include:

            - "chr1", "chr2", ..., "chrX", "chrY"
            - "Chr1", "Chr2", ..., "ChrX", "ChrY"
            - "1", "2", ..., "X", "Y"

        start_col : str, default "start"
            The column name of the start position column.
        end_col : str, default "end"
            The column name of the end position column. If `end_col` is the same as `start_col`, the result will include "start" and "end" columns that are the same.
        copy_number_col : str, default "copy_number"
            The column name of the chromosome column. The values can be integers or decimals.
        """
        self.data: PolarsFrame = self.copy_number_preprocess(
            copy_number_data=copy_number_data,
            chr_col=chr_col,
            start_col=start_col,
            end_col=end_col,
            copy_number_col=copy_number_col,
        )

    @classmethod
    def region_consistency_check(cls, copy_number_x: Self, copy_number_y: Self) -> bool:
        """
        Check if the regions (genome windows or chromosome names) of two copy number data are consistent.

        Parameters
        ----------
        copy_number_x : Self
            A copy number data.
        copy_number_y : Self
            A copy number data.

        Returns
        -------
        bool
            If the regions of two copy number data are consistent, return True. Otherwise, return False.
        """
        x_region: PolarsFrame = copy_number_x.data.select(
            pl.col(["chr", "start", "end"])
        )
        x_region_df: pl.DataFrame = (
            x_region.collect() if isinstance(x_region, pl.LazyFrame) else x_region
        )
        y_region: PolarsFrame = copy_number_y.data.select(
            pl.col(["chr", "start", "end"])
        )
        y_region_df: pl.DataFrame = (
            y_region.collect() if isinstance(y_region, pl.LazyFrame) else y_region
        )
        return x_region_df.equals(y_region_df)

    @classmethod
    def accuracy_score(cls, copy_number_true: Self, copy_number_pred: Self) -> float:
        """
        Calculate the accuracy score of two copy number data.

        Parameters
        ----------
        copy_number_true : Self
            A copy number data as reference.
        copy_number_pred : Self
            A copy number data to be evaluated.

        Returns
        -------
        float
            The accuracy score of two copy number data.
        """
        region_cols: list[str] = ["chr", "start", "end"]
        required_cols: list[str] = region_cols + ["integer_copy_number"]
        copy_number_true_data: PolarsFrame = copy_number_true.data.select(
            required_cols
        ).rename({"integer_copy_number": "integer_copy_number_true"})
        copy_number_pred_data: PolarsFrame = copy_number_pred.data.select(
            required_cols
        ).rename({"integer_copy_number": "integer_copy_number_pred"})
        joined_copy_number_data: PolarsFrame = copy_number_true_data.join(
            copy_number_pred_data, on=region_cols, how="inner", coalesce=True
        ).select(pl.col(["integer_copy_number_true", "integer_copy_number_pred"]))
        joined_copy_number_df: pl.DataFrame = (
            joined_copy_number_data.collect()
            if isinstance(joined_copy_number_data, pl.LazyFrame)
            else joined_copy_number_data
        )
        result: float = float(
            metrics.accuracy_score(
                joined_copy_number_df["integer_copy_number_true"],
                joined_copy_number_df["integer_copy_number_pred"],
            )
        )
        return result

    @classmethod
    def recall_score(
        cls,
        copy_number_true: Self,
        copy_number_pred: Self,
        average: Literal["micro", "samples", "weighted", "macro"] | None = "macro",
    ) -> float:
        """
        Calculate the recall score of two copy number data.

        Parameters
        ----------
        copy_number_true : Self
            A copy number data as reference.
        copy_number_pred : Self
            A copy number data to be evaluated.
        average : Literal["micro", "samples", "weighted", "macro"] | None, default  "macro"
            If ``None``, the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the data:

            ``'micro'``:
                Calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label).
            ``'samples'``:
                Calculate metrics for each instance, and find their average.

            Will be ignored when ``y_true`` is binary.

        Returns
        -------
        float
            The recall score of two copy number data.
        """
        region_cols: list[str] = ["chr", "start", "end"]
        required_cols: list[str] = region_cols + ["integer_copy_number"]
        copy_number_true_data: PolarsFrame = copy_number_true.data.select(
            required_cols
        ).rename({"integer_copy_number": "integer_copy_number_true"})
        copy_number_pred_data: PolarsFrame = copy_number_pred.data.select(
            required_cols
        ).rename({"integer_copy_number": "integer_copy_number_pred"})
        joined_copy_number_data: PolarsFrame = copy_number_true_data.join(
            copy_number_pred_data, on=region_cols, how="inner", coalesce=True
        ).select(pl.col(["integer_copy_number_true", "integer_copy_number_pred"]))
        joined_copy_number_df: pl.DataFrame = (
            joined_copy_number_data.collect()
            if isinstance(joined_copy_number_data, pl.LazyFrame)
            else joined_copy_number_data
        )
        result: float = float(
            metrics.recall_score(
                joined_copy_number_df["integer_copy_number_true"],
                joined_copy_number_df["integer_copy_number_pred"],
                average=average,
            )
        )
        return result

    @classmethod
    def precision_score(
        cls,
        copy_number_true: Self,
        copy_number_pred: Self,
        average: Literal["micro", "samples", "weighted", "macro"] | None = "macro",
    ) -> float:
        """
        Calculate the precision score of two copy number data.

        Parameters
        ----------
        copy_number_true : Self
            A copy number data as reference.
        copy_number_pred : Self
            A copy number data to be evaluated.
        average : Literal["micro", "samples", "weighted", "macro"] | None, default  "macro"
            If ``None``, the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the data:

            ``'micro'``:
                Calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label).
            ``'samples'``:
                Calculate metrics for each instance, and find their average.

            Will be ignored when ``y_true`` is binary.

        Returns
        -------
        float
            The precision score of two copy number data.
        """
        region_cols: list[str] = ["chr", "start", "end"]
        required_cols: list[str] = region_cols + ["integer_copy_number"]
        copy_number_true_data: PolarsFrame = copy_number_true.data.select(
            required_cols
        ).rename({"integer_copy_number": "integer_copy_number_true"})
        copy_number_pred_data: PolarsFrame = copy_number_pred.data.select(
            required_cols
        ).rename({"integer_copy_number": "integer_copy_number_pred"})
        joined_copy_number_data: PolarsFrame = copy_number_true_data.join(
            copy_number_pred_data, on=region_cols, how="inner", coalesce=True
        ).select(pl.col(["integer_copy_number_true", "integer_copy_number_pred"]))
        joined_copy_number_df: pl.DataFrame = (
            joined_copy_number_data.collect()
            if isinstance(joined_copy_number_data, pl.LazyFrame)
            else joined_copy_number_data
        )
        result: float = float(
            metrics.precision_score(
                joined_copy_number_df["integer_copy_number_true"],
                joined_copy_number_df["integer_copy_number_pred"],
                average=average,
            )
        )
        return result

    @classmethod
    def difference_std(cls, copy_number_true: Self, copy_number_pred: Self) -> float:
        """
        Calculate the standard deviation of the difference between two copy number data.

        Parameters
        ----------
        copy_number_true : Self
            A copy number data as reference.
        copy_number_pred : Self
            A copy number data to be evaluated.

        Returns
        -------
        float
            The standard deviation of the difference between two copy number data.
        """
        region_cols: list[str] = ["chr", "start", "end"]
        required_cols: list[str] = region_cols + ["copy_number"]
        copy_number_true_data: PolarsFrame = copy_number_true.data.select(
            required_cols
        ).rename({"copy_number": "copy_number_true"})
        copy_number_pred_data: PolarsFrame = copy_number_pred.data.select(
            required_cols
        ).rename({"copy_number": "copy_number_pred"})
        joined_copy_number_data: PolarsFrame = copy_number_true_data.join(
            copy_number_pred_data, on=region_cols, how="inner", coalesce=True
        ).select(pl.col(["copy_number_true", "copy_number_pred"]))
        joined_copy_number_df: pl.DataFrame = (
            joined_copy_number_data.collect()
            if isinstance(joined_copy_number_data, pl.LazyFrame)
            else joined_copy_number_data
        )
        result: float = float(
            (
                joined_copy_number_df["copy_number_pred"]
                - joined_copy_number_df["copy_number_true"]
            ).std()
        )
        return result

    @classmethod
    def mean(cls, copy_numbers: Sequence[Self]) -> Self:
        """
        Calculate the mean copy number of multiple copy number data. `CopyNumberWindow.region_consistency_check()` will ber performed before calculating the mean.

        Parameters
        ----------
        copy_numbers : Iterable[CopyNumberWindow]
            Multiple copy number data.

        Returns
        -------
        CopyNumberWindow
            The mean of multiple copy number data as a `CopyNumberWindow` object.
        """
        data_number: int = len(copy_numbers)
        for i in range(data_number - 1):
            if not cls.region_consistency_check(copy_numbers[i], copy_numbers[i + 1]):
                raise ValueError(f"The regions of copy number data are inconsistent. The inconsistency occurred between index {i} and {i+1}.")

        def copy_numbers_get_cn(i: int) -> pl.Series:
            if isinstance(copy_numbers[i].data, pl.LazyFrame):
                return (
                    copy_numbers[i]
                    .data.select(["copy_number"])
                    .collect()
                    .get_column("copy_number")
                )
            else:
                return (
                    copy_numbers[i]
                    .data.select(["copy_number"])
                    .get_column("copy_number")
                )

        copy_number_mean: pl.Series = copy_numbers_get_cn(0) / data_number

        for i in range(1, data_number):
            copy_number_mean = copy_number_mean + (copy_numbers_get_cn(i) / data_number)

        result_df: pl.DataFrame
        if isinstance(copy_numbers[0].data, pl.LazyFrame):
            result_df = (
                copy_numbers[0]
                .data.with_columns(
                    copy_number=copy_number_mean,
                    copy_number_integer=copy_number_mean.round(0),
                )
                .collect()
            )
        else:
            result_df = copy_numbers[0].data.with_columns(copy_number=copy_number_mean)

        result = CopyNumberWindow(
            copy_number_data=result_df,
            chr_col="chr",
            start_col="start",
            end_col="end",
            copy_number_col="copy_number",
        )
        return result

    @classmethod
    def correlation_coefficient(
        cls,
        copy_number_x: Self,
        copy_number_y: Self,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> float:
        """
        Calculate the correlation coefficient between two copy number data. `CopyNumber.region_consistency_check()` will be performed before calculating the correlation coefficient.

        Parameters
        ----------
        copy_number_x : CopyNumber
            A copy number data.
        copy_number_y : CopyNumber
            A copy number data.
        method : Literal["pearson", "spearman"], default "pearson"
            The method to calculate the correlation coefficient.

        Returns
        -------
        float
            Correlation coefficient between two copy number data.
        """
        if (
            cls.region_consistency_check(
                copy_number_x=copy_number_x, copy_number_y=copy_number_y
            )
            is False
        ):
            raise ValueError("The regions of copy number data are inconsistent.")

        required_cols: list[str] = ["copy_number"]
        copy_number_x_data: PolarsFrame = copy_number_x.data.select(required_cols)
        copy_number_y_data: PolarsFrame = copy_number_y.data.select(required_cols)
        copy_number_x_df: pl.DataFrame = (
            copy_number_x_data.collect()
            if isinstance(copy_number_x_data, pl.LazyFrame)
            else copy_number_x_data
        )
        copy_number_y_df: pl.DataFrame = (
            copy_number_y_data.collect()
            if isinstance(copy_number_y_data, pl.LazyFrame)
            else copy_number_y_data
        )
        result_df: pl.DataFrame = (
            copy_number_x_df.rename({"copy_number": "copy_number_x"})
            .with_columns(copy_number_y=copy_number_y_df["copy_number"])
            .select(
                pl.corr(
                    "copy_number_x",
                    "copy_number_y",
                    propagate_nans=True,  # any `NaN` will lead to `NaN` output
                    method=method,
                )
            )
        )
        result: float = float(result_df[0, 0])
        return result

    def manhattan_plot_preprocess(self) -> PolarsFrame:
        """Preprocess the data for manhattan plot."""
        manhattan_data: PolarsFrame = (
            self.data.with_columns(
                position=((pl.col("start") + pl.col("end")) // 2).cast(pl.Int64),
            )
            .filter(pl.col("chr").is_in(self._chromosome_names))
            .with_columns(
                chr_number=pl.col("chr")
                .replace(
                    self._chromosome_rename
                )
                .cast(pl.Int8)
            )
            .sort(["chr_number", "position"], descending=False)
        )
        return manhattan_data

    def manhattan_plot(self) -> ManhattanPlot:
        """Draw manhattan plot of copy number."""
        manhattan_data: PolarsFrame = self.manhattan_plot_preprocess()
        manhattan_df: pl.DataFrame = (
            manhattan_data.collect()
            if isinstance(manhattan_data, pl.LazyFrame)
            else manhattan_data
        )
        manhattan_plot: go.Figure = (
            dash_bio.ManhattanPlot(
                dataframe=manhattan_df.to_pandas(),
                chrm="chr_number",
                bp="position",
                p="copy_number",
                xlabel="Chromosome",
                ylabel="Copy number",
                snp=None,
                gene=None,
                logp=False,
                highlight=False,
                genomewideline_value=False,
                suggestiveline_value=False,
                showlegend=False,
                point_size=4,
            )
            .update_layout(template="plotly_white", width=1600, height=400)
            .update_yaxes(range=[0, (max_copy_number := 4)])
        )
        if (chr_count := len(manhattan_plot.data)) not in [
            len(self._chromosome_names),
            len(self._chromosome_names) - 1,
        ]:
            raise ValueError(
                f"Unexpected number of chromosomes in the plot: {len(manhattan_plot.data)}."
            )

        # customise colours
        point_colours: list[str] = [
            "rgba(99,110,250,0.4)",
            "rgba(239,85,59,0.4)",
            "rgba(0,204,150,0.4)",
            "rgba(171,99,250,0.4)",
            "rgba(255,161,90,0.4)",
            "rgba(25,211,243,0.4)",
            "rgba(255,102,146,0.4)",
            "rgba(182,232,128,0.4)",
            "rgba(255,151,255,0.4)",
            "rgba(254,203,82,0.4)",
        ]
        chr_colours: list[str] = [
            point_colours[x % len(point_colours)] for x in range(chr_count)
        ]
        for i in range(chr_count):
            manhattan_plot.data[i].marker.color = chr_colours[i]

        # x-axis tick customise
        curr_chr_index: int = 0
        curr_chr_x_min: int = manhattan_plot.data[curr_chr_index].x.min()
        curr_chr_x_max: int = manhattan_plot.data[curr_chr_index].x.max()
        curr_chr_x_med: int = (curr_chr_x_min + curr_chr_x_max) // 2
        tickvals: list[int] = [curr_chr_x_med]
        ticktext: list[str] = [self._chromosome_names[curr_chr_index]]
        minor_tickvals: list[int] = [curr_chr_x_min, curr_chr_x_max]

        # x-coordinate - position difference
        last_base = 0
        chr_pos_diff: list[list[str, int]] = [
            [self._chromosome_names[curr_chr_index], last_base]
        ]
        for curr_chr_index in range(1, chr_count):
            # x-axis tick customise
            last_chr_copy_number_df: pl.DataFrame = manhattan_df.filter(
                pl.col("chr") == self._chromosome_names[curr_chr_index - 1]
            ).select(pl.col(["chr", "position"]))
            curr_chr_copy_number_df: pl.DataFrame = manhattan_df.filter(
                pl.col("chr") == self._chromosome_names[curr_chr_index]
            ).select(pl.col(["chr", "position"]))
            curr_chr_x_min = (
                manhattan_plot.data[curr_chr_index].x.min() + curr_chr_x_max
            ) // 2
            curr_chr_x_max = manhattan_plot.data[curr_chr_index].x.max()
            curr_chr_x_med = (curr_chr_x_min + curr_chr_x_max) // 2
            tickvals.append(curr_chr_x_med)
            ticktext.append(self._chromosome_names[curr_chr_index])
            minor_tickvals.append(curr_chr_x_max)

            # x-coordinate - position difference
            last_base += last_chr_copy_number_df[-1, "position"]
            curr_chr_bp_array: np.ndarray = (
                curr_chr_copy_number_df["position"].cast(int).to_numpy()
            )
            curr_chr_x_array: np.ndarray = manhattan_plot.data[curr_chr_index].x
            curr_chr_x_bp_diff_array: np.ndarray = curr_chr_x_array - curr_chr_bp_array

            if np.unique(curr_chr_x_bp_diff_array).shape[0] != 1:
                raise ValueError(
                    f'Chromosome "{self._chromosome_names[curr_chr_index]}" does not have a uniform "x-coordinate - position" difference.'
                )
            elif np.unique(curr_chr_x_bp_diff_array)[0] != last_base:
                raise ValueError(
                    f'Chromosome "{self._chromosome_names[curr_chr_index]}" does not have expected "x-coordinate - position" difference. Expected: {last_base}, Actual: {curr_chr_x_bp_diff_array[0]}. Check the version of `dash_bio`.'
                )

            chr_pos_diff.append([self._chromosome_names[curr_chr_index], last_base])

        manhattan_plot.update_xaxes(
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=False,
            ticklen=0,
            minor_tickvals=minor_tickvals,
            minor_ticklen=0,
            minor_showgrid=True,
            minor_gridcolor="#EBF0F8",
        )
        manhattan_position_converter: pl.DataFrame = pl.DataFrame(
            data={
                "chr": [x[0] for x in chr_pos_diff],
                "position_diff": [x[1] for x in chr_pos_diff],
            },
            schema=pl.Schema({"chr": pl.String, "position_diff": pl.Int64}),
            strict=True,
            orient="row",
        )

        return ManhattanPlot(
            figure=manhattan_plot, position_converter=manhattan_position_converter
        )
