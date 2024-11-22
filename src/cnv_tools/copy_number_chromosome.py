"""
copy_number_chromosome

======================

Copy number data of chromosomes.
"""

from typing import Literal, Self, Sequence

import polars as pl
from sklearn import metrics

from cnv_tools._typing import PolarsFrame
from cnv_tools.copy_number import CopyNumber
from cnv_tools.manhattan_plot import ManhattanPlot


class CopyNumberChromosome(CopyNumber):
    """
    Copy number information of chromosomes.

    Attributes
    ----------
    data : PolarsFrame
        Copy number data of genome windows. A polars DataFrame / LazyFrame containing the following columns:

        - "chr": polars.String
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
    """

    def __init__(
        self,
        copy_number_data: PolarsFrame,
        chr_col: str = "chr",
        start_col: str | None = "start",
        end_col: str | None = "end",
        copy_number_col: str = "copy_number",
    ) -> None:
        """
        Create a `CopyNumberChromosome`.

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
        start_col : str | None, default "start"
            The column name of the start position column. Either `start_col` or `end_col` is `None` means there is no start column, e.g. for chromosome copy number data.
        end_col : str | None, default "end"
            The column name of the end position column. If `end_col` is the same as `start_col`, the result will include "start" and "end" columns that are the same. Either `start_col` or `end_col` is `None` means there is no start column, e.g. for chromosome copy number data.
        copy_number_col : str, default "copy_number"
            The column name of the copy number column. The values can be integers or decimals.
        """
        self.data: PolarsFrame = self.copy_number_preprocess(
            copy_number_data,
            chr_col=chr_col,
            start_col=start_col,
            end_col=end_col,
            copy_number_col=copy_number_col,
        )
        self.data = self.chromosome_copy_number(self.data)
        self.manhattan_data: PolarsFrame | None = None

    @classmethod
    def region_consistency_check(cls, copy_number_x: Self, copy_number_y: Self) -> bool:
        """Check if the regions (genome windows or chromosome names) of two copy number data are consistent."""
        x_region: PolarsFrame = copy_number_x.data.select(pl.col(["chr"]))
        x_region_df: pl.DataFrame = (
            x_region.collect() if isinstance(x_region, pl.LazyFrame) else x_region
        )
        y_region: PolarsFrame = copy_number_y.data.select(pl.col(["chr"]))
        y_region_df: pl.DataFrame = (
            y_region.collect() if isinstance(y_region, pl.LazyFrame) else y_region
        )
        return x_region_df.equals(y_region_df)

    @classmethod
    def accuracy_score(cls, copy_number_true: Self, copy_number_pred: Self) -> float:
        region_cols: list[str] = ["chr"]
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
        region_cols: list[str] = ["chr"]
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
        region_cols: list[str] = ["chr"]
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
        region_cols: list[str] = ["chr"]
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
                raise ValueError("The regions of copy number data are inconsistent.")

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
                .data.with_columns(copy_number=copy_number_mean)
                .collect()
            )
        else:
            result_df = copy_numbers[0].data.with_columns(
                copy_number=copy_number_mean,
                copy_number_integer=copy_number_mean.round(0),
            )

        result = CopyNumberChromosome(
            copy_number_data=result_df,
            chr_col="chr",
            start_col=None,
            end_col=None,
            copy_number_col="copy_number",
        )
        return result
