"""
cnv

=====================

A module for copy number variation analysis.
"""

from typing import Self

from cnv_tools._typing import PolarsFrame
from cnv_tools.copy_number import CopyNumber


class Cnv(CopyNumber):
    """
    Copy number variation information.

    Attributes
    ----------
    data : PolarsFrame
        A polars DataFrame / LazyFrame containing the following columns:

        - "chr": polars.String
        - "start": polars.Int64
        - "end": polars.Int64
        - "variation_type": polars.Float32

        The "chr" column are like "1", "2", ..., "X", "Y". The rows are sorted by the "chr" and "start" columns. The "chr" column order are like "1", "2", ..., "22", "X", "Y". Other columns will not be modified.
    """

    # TODO: the metrics like accuracy score are not implemented. The definition of TP needs further consideration. Reference: <https://www.mdpi.com/2072-6694/13/24/6283>

    def __init__(
        self,
        cnv_data: PolarsFrame,
        chr_col: str = "chr",
        start_col: str = "start",
        end_col: str = "end",
        cnv_type_col: str = "cnv_type",
        gain_value: str = "gain",
        loss_value: str = "loss",
    ) -> None:
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
        self.data = self.cnv_preprocess(
            cnv_data=cnv_data,
            chr_col=chr_col,
            start_col=start_col,
            end_col=end_col,
            cnv_type_col=cnv_type_col,
            gain_value=gain_value,
            loss_value=loss_value,
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
        # TODO
        raise NotImplementedError

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
        # TODO
        raise NotImplementedError

    @classmethod
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
        # TODO
        raise NotImplementedError

    @classmethod
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
        # TODO
        raise NotImplementedError

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
        # TODO
        raise NotImplementedError
