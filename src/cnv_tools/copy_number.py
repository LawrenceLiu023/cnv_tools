"""
copy_number

===========

Classes for copy number data. This module defines the basic data structure for copy number data and further analysis.
"""

from abc import ABC, abstractmethod
from typing import Self

import polars as pl

from cnv_tools._typing import PolarsFrame


class CopyNumber(ABC):
    """A class for copy number data."""

    CHROMOSOME_NAMES: list[str] = [str(x) for x in range(1, 23)] + ["X", "Y"]

    @abstractmethod
    def __init__(self) -> None:
        self.data: PolarsFrame
        pass

    @classmethod
    def copy_number_preprocess(
        cls,
        copy_number_data: PolarsFrame,
        chr_col: str = "chr",
        start_col: str = "start",
        end_col: str = "end",
        copy_number_col: str = "copy_number",
    ) -> PolarsFrame:
        """
        Preprocess the DataFrame containing genome copy number information for further analysis. The columns are renamed, and the column data types are reset.

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

        Returns
        -------
        preprocessed_copy_number_data : PolarsFrame
            A polars DataFrame / LazyFrame containing the following columns:

            - "chr": polars.String
            - "start": polars.Int64
            - "end": polars.Int64
            - "copy_number": polars.Float32
            - "integer_copy_number": polars.Int32

            The "chr" column are like "1", "2", ..., "X", "Y". The rows are sorted by the "chr" and "start" columns. The "chr" column order are like "1", "2", ..., "22", "X", "Y". Other columns will not be modified.
        """
        copy_number_lf: pl.LazyFrame
        if isinstance(copy_number_data, pl.DataFrame):
            copy_number_lf = copy_number_data.lazy()
        elif isinstance(copy_number_data, pl.LazyFrame):
            copy_number_lf = copy_number_data
        else:
            raise TypeError(
                f"`copy_number_data` must be `polars.DataFrame` or `polars.LazyFrame`. Got {type(copy_number_data)}"
            )

        # chromosome
        # Rename the column as "chr". Format: 1, 2, 3, ..., 22, X, Y
        chr_col_unique_series: pl.Series = copy_number_lf.select(
            pl.col(chr_col).unique().alias("chr")
        ).collect()["chr"]
        if chr_col_unique_series.str.starts_with("chr").all():
            copy_number_lf = copy_number_lf.with_columns(
                pl.col(chr_col).str.replace("chr", "").alias(chr_col)
            ).rename({chr_col: "chr"})
        if chr_col_unique_series.str.starts_with("Chr").all():
            copy_number_lf = copy_number_lf.with_columns(
                pl.col(chr_col).str.replace("Chr", "").alias(chr_col)
            ).rename({chr_col: "chr"})
        else:
            copy_number_lf = copy_number_lf.rename({chr_col: "chr"})

        # start
        copy_number_lf = copy_number_lf.rename({start_col: "start"})

        # end
        if end_col == start_col:
            copy_number_lf = copy_number_lf.with_columns(end=pl.col("start"))
        else:
            copy_number_lf = copy_number_lf.rename({end_col: "end"})

        # copy number
        copy_number_lf = copy_number_lf.rename(
            {copy_number_col: "copy_number"}
        ).with_columns(
            pl.col("copy_number").round(0).cast(pl.Int32).alias("integer_copy_number")
        )

        # data type
        copy_number_lf = copy_number_lf.cast(
            {
                "chr": pl.String,
                "start": pl.Int64,
                "end": pl.Int64,
                "copy_number": pl.Float32,
                "integer_copy_number": pl.Int32,
            }
        )

        copy_number_lf = copy_number_lf.filter(
            pl.col("chr").is_in(cls.CHROMOSOME_NAMES)
        ).sort(
            [
                pl.col("chr")
                .replace(
                    {
                        "X": "23",
                        "Y": "24",
                    }
                )
                .cast(pl.Int8),
                pl.col("start"),
            ],
            descending=False,
        )

        preprocessed_copy_number_data: PolarsFrame
        if isinstance(copy_number_data, pl.DataFrame):
            preprocessed_copy_number_data = copy_number_lf.collect()
        else:
            preprocessed_copy_number_data = copy_number_lf
        return preprocessed_copy_number_data

    @staticmethod
    def chromosome_copy_number(
        copy_number_data: PolarsFrame,
    ) -> PolarsFrame:
        """
        Calculate the copy number of each chromosome.

        Parameters
        ----------
        copy_number_data : PolarsFrame
            A polars DataFrame / LazyFrame that contains copy number information of each genome window. It should have been preprocessed by `copy_number_preprocess`.

        Returns
        -------
        chromosome_copy_number : polars.DataFrame | polars.LazyFrame
            A polars DataFrame or LazyFrame that contains the following columns:

            - "chr": polars.String
            - "copy_number": polars.Float32
            - "integer_copy_number": pl.Int32

            Other columns will not be modified.
        """
        chromosome_copy_number: PolarsFrame = (
            copy_number_data.select(pl.col(["chr", "copy_number"]))
            .group_by("chr", maintain_order=True)
            .mean()
        ).with_columns(
            pl.col("copy_number").round(0).cast(pl.Int32).alias("integer_copy_number")
        )
        return chromosome_copy_number

    @classmethod
    def cnv_preprocess(
        cls,
        cnv_data: PolarsFrame,
        chr_col: str = "chr",
        start_col: str = "start",
        end_col: str = "end",
        cnv_type_col: str = "cnv_type",
        gain_value: str = "gain",
        loss_value: str = "loss",
    ) -> PolarsFrame:
        """
        Preprocess the copy number variation data.

        Parameters
        ----------
        cnv : PolarsFrame
            A polars DataFrame / LazyFrame containing the copy number variation data.
        chr_col : str, default "chr"
            The column name of chromosome.
        start_col : str, default "start"
            The column name of start position.
        end_col : str, default "end"
            The column name of end position.
        cnv_type_col : str, default "cnv_type"
            The column name of CNV type.
        gain_value : str, default "gain"
            The string represents type of gain.
        loss_value : str, default "loss"
            The string represents type of loss.

        Returns
        -------
        data : PolarsFrame
            A polars DataFrame / LazyFrame containing the following columns:

            - "chr": polars.String
            - "start": polars.Int64
            - "end": polars.Int64
            - "cnv_type": polars.Categorical(ordering="lexical")

            The "chr" column are like "1", "2", ..., "X", "Y". The rows are sorted by the "chr" and "start" columns. The "chr" column order are like "1", "2", ..., "22", "X", "Y". Other columns will not be modified.
        """
        cnv_lf: pl.LazyFrame
        if isinstance(cnv_data, pl.DataFrame):
            cnv_lf = cnv_data.lazy()
        elif isinstance(cnv_data, pl.LazyFrame):
            cnv_lf = cnv_data
        else:
            raise TypeError(
                f"`cnv_data` must be `polars.DataFrame` or `polars.LazyFrame`. Got {type(cnv_data)}"
            )

        # chromosome
        # Rename the column as "chr". Format: 1, 2, 3, ..., 22, X, Y
        chr_col_unique_series: pl.Series = cnv_lf.select(
            pl.col(chr_col).unique().alias("chr")
        ).collect()["chr"]
        if chr_col_unique_series.str.starts_with("chr").all():
            cnv_lf = cnv_lf.with_columns(
                pl.col(chr_col).str.replace("chr", "").alias(chr_col)
            ).rename({chr_col: "chr"})
        if chr_col_unique_series.str.starts_with("Chr").all():
            cnv_lf = cnv_lf.with_columns(
                pl.col(chr_col).str.replace("Chr", "").alias(chr_col)
            ).rename({chr_col: "chr"})
        else:
            cnv_lf = cnv_lf.rename({chr_col: "chr"})

        # start
        cnv_lf = cnv_lf.rename({start_col: "start"})

        # end
        if end_col == start_col:
            cnv_lf = cnv_lf.with_columns(end=pl.col("start"))
        else:
            cnv_lf = cnv_lf.rename({end_col: "end"})

        # CNV type
        cnv_lf = cnv_lf.rename({cnv_type_col: "cnv_type"}).with_columns(
            pl.col("cnv_type").replace_strict({gain_value: "gain", loss_value: "loss"})
        )
        cnv_lf = cnv_lf.cast(
            {
                "chr": pl.String,
                "start": pl.Int64,
                "end": pl.Int64,
                "cnv_type": pl.Categorical(ordering="lexical"),
            }
        )
        cnv_lf = cnv_lf.filter(pl.col("chr").is_in(cls.CHROMOSOME_NAMES)).sort(
            [
                pl.col("chr")
                .replace(
                    {
                        "X": "23",
                        "Y": "24",
                    }
                )
                .cast(pl.Int8),
                pl.col("start"),
            ],
            descending=False,
        )
        preprocessed_cnv_data: PolarsFrame
        if isinstance(cnv_data, pl.DataFrame):
            preprocessed_cnv_data = cnv_lf.collect()
        else:
            preprocessed_cnv_data = cnv_lf
        return preprocessed_cnv_data

    @classmethod
    @abstractmethod
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
        pass

    @classmethod
    @abstractmethod
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
        pass

    @classmethod
    @abstractmethod
    def recall_score(cls, copy_number_true: Self, copy_number_pred: Self) -> float:
        """
        Calculate the recall score of two copy number data.

        Parameters
        ----------
        copy_number_true : Self
            A copy number data as reference.
        copy_number_pred : Self
            A copy number data to be evaluated.

        Returns
        -------
        float
            The recall score of two copy number data.
        """
        pass

    @classmethod
    @abstractmethod
    def precision_score(cls, copy_number_true: Self, copy_number_pred: Self) -> float:
        """
        Calculate the precision score of two copy number data.

        Parameters
        ----------
        copy_number_true : Self
            A copy number data as reference.
        copy_number_pred : Self
            A copy number data to be evaluated.

        Returns
        -------
        float
            The precision score of two copy number data.
        """
        pass

    @classmethod
    @abstractmethod
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
        pass


def region_consistency_check(
    copy_number_x: CopyNumber, copy_number_y: CopyNumber
) -> bool:
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
    return copy_number_x.region_consistency_check(
        copy_number_x=copy_number_x, copy_number_y=copy_number_y
    )


def accuracy_score(copy_number_true: CopyNumber, copy_number_pred: CopyNumber) -> float:
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
    return copy_number_true.accuracy_score(
        copy_number_true=copy_number_true, copy_number_pred=copy_number_pred
    )


def recall_score(copy_number_true: CopyNumber, copy_number_pred: CopyNumber) -> float:
    """
    Calculate the recall score of two copy number data.

    Parameters
    ----------
    copy_number_true : Self
        A copy number data as reference.
    copy_number_pred : Self
        A copy number data to be evaluated.

    Returns
    -------
    float
        The recall score of two copy number data.
    """
    return copy_number_true.recall_score(
        copy_number_true=copy_number_true, copy_number_pred=copy_number_pred
    )


def precision_score(
    copy_number_true: CopyNumber, copy_number_pred: CopyNumber
) -> float:
    """
    Calculate the precision score of two copy number data.

    Parameters
    ----------
    copy_number_true : Self
        A copy number data as reference.
    copy_number_pred : Self
        A copy number data to be evaluated.

    Returns
    -------
    float
        The precision score of two copy number data.
    """
    return copy_number_true.precision_score(
        copy_number_true=copy_number_true, copy_number_pred=copy_number_pred
    )


def difference_std(copy_number_true: CopyNumber, copy_number_pred: CopyNumber) -> float:
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
    return copy_number_true.difference_std(
        copy_number_true=copy_number_true, copy_number_pred=copy_number_pred
    )
