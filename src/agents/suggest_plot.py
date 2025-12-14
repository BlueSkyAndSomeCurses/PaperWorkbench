import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from langchain_openai import ChatOpenAI

from src.agents.states import AgentState, State
from src.agents.suggest_plot_base import PlotSuggester


# @dataclass
# class PlotSuggestion:
#     """Structured plot suggestion with metadata"""
#     id: str
#     description: str
#     code: str
#     rationale: str
#     approved: bool = False
#     filename_base: Optional[Path] = None
#     figure: Optional[plt.Figure] = None


SUPPORTED_TABLE_FORMATS = ['.csv', '.xlsx', '.xls', '.tsv', '.parquet']


def extract_table_data(file_path: Path, max_rows: int = 5) -> tuple[str, str, str]:
    """
    Extract data preview from table files.
    
    Args:
        file_path: Path to CSV/Excel file
        max_rows: Number of rows to preview
        
    Returns:
        Tuple of (data_preview, columns_desc, shape_str)
    """
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.tsv':
            df = pd.read_csv(file_path, sep='\t')
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Generate preview
        data_preview = df.head(max_rows).to_string()
        columns_desc = ", ".join(df.columns.tolist())
        shape_str = f"{df.shape[0]} rows × {df.shape[1]} columns"
        
        return data_preview, columns_desc, shape_str
        
    except Exception as e:
        logging.error(f"Failed to extract data from {file_path}: {e}")
        return "Failed to load data", "Unknown", "Unknown"

