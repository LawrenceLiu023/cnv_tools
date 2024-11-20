from typing import TypeVar

from polars import DataFrame, LazyFrame

# Generic type for polars DataFrame or LazyFrame.
# Methods are designed to output the same type as input.
PolarsFrame = TypeVar("PolarsFrame", bound=DataFrame | LazyFrame)
