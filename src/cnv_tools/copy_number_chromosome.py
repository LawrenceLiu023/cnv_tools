"""
copy_number_chromosome

======================

Copy number data of chromosomes.
"""

from typing import Self

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
    recall_score(copy_number_true: CopyNumberWindow, copy_number_pred: CopyNumberWindow) -> float
        Calculate the recall score of two copy number data.
    precision_score(copy_number_true: CopyNumberWindow, copy_number_pred: CopyNumberWindow) -> float
        Calculate the precision score of two copy number data.
    difference_std(copy_number_true: CopyNumberWindow, copy_number_pred: CopyNumberWindow) -> float
        Calculate the standard deviation of the difference between two copy number data.
    """

    def __init__(
        self,
        copy_number_data: PolarsFrame,
        chr_col="chr",
        start_col="start",
        end_col="end",
        copy_number_col="copy_number",
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
        start_col : str, default "start"
            The column name of the start position column.
        end_col : str, default "end"
            The column name of the end position column. If `end_col` is the same as `start_col`, the result will include "start" and "end" columns that are the same.
        copy_number_col : str, default "copy_number"
            The column name of the chromosome column. The values can be integers or decimals.
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
    def recall_score(cls, copy_number_true: Self, copy_number_pred: Self) -> float:
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
            )
        )
        return result

    @classmethod
    def precision_score(cls, copy_number_true: Self, copy_number_pred: Self) -> float:
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
