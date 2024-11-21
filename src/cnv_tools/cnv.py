"""
cnv

=====================

A module for copy number variation analysis.
"""

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
        cnv_type: str = "cnv_type",
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
        cnv_type : str, default "cnv_type"
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
            cnv_type=cnv_type,
            gain_value=gain_value,
            loss_value=loss_value,
        )
