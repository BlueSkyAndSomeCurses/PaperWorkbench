from pathlib import Path

import polars as pl

from src.utils.constants import SUPPORTED_TABLE_FORMATS


def extract_table_data(file_path: Path) -> tuple[str, dict, str]:
    """
    returns the first 6 rows of a table file as a string representation. Returns column names, data types and shape.
    """

    if file_path.suffix in SUPPORTED_TABLE_FORMATS:
        match file_path.suffix:
            case ".csv":
                lazy_table = pl.scan_csv(file_path)
            case ".parquet":
                lazy_table = pl.scan_parquet(file_path)
            case _:
                raise ValueError("Unsupported table format.")

        return (
            lazy_table.head(6).collect().to_init_repr(),
            lazy_table.collect_schema().to_python(),
            lazy_table.select(pl.all().approx_n_unique()).collect().to_init_repr(),
        )

    return "Unsupported table type", {}, "Unsupported table type"


def handle_file_reading_for_model(file_path: Path) -> str:
    if file_path.suffix in SUPPORTED_TABLE_FORMATS:
        return extract_table_data(file_path)[0]

    with file_path.open("r", encoding="utf-8") as f:
        return f.read()
