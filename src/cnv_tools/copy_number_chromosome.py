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
    """Copy number information of chromosomes."""

    def __init__(
        self,
        copy_number: PolarsFrame,
        chr_col="chr",
        start_col="start",
        end_col="end",
        copy_number_col="copy_number",
    ) -> None:
        self.data: PolarsFrame = self.copy_number_preprocess(
            copy_number,
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
        result: float = metrics.accuracy_score(
            joined_copy_number_df["integer_copy_number_true"],
            joined_copy_number_df["integer_copy_number_pred"],
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
        result: float = metrics.recall_score(
            joined_copy_number_df["integer_copy_number_true"],
            joined_copy_number_df["integer_copy_number_pred"],
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
        result: float = metrics.precision_score(
            joined_copy_number_df["integer_copy_number_true"],
            joined_copy_number_df["integer_copy_number_pred"],
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
        result: float = (
            joined_copy_number_df["copy_number_pred"]
            - joined_copy_number_df["copy_number_true"]
        ).std()
        return result
