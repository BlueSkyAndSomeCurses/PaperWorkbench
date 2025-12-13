from dataclasses import dataclass
from typing import Optional, List
import uuid
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from langchain_openai import ChatOpenAI
from src.agents.states import AgentState, State
from src.agents.suggest_plot_base import PlotSuggester


@dataclass
class PlotSuggestion:
    """Structured plot suggestion with metadata"""
    id: str
    description: str
    code: str
    rationale: str
    approved: bool = False
    filename_base: Optional[Path] = None
    figure: Optional[plt.Figure] = None


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
        # Load file based on extension
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


class PlotSuggestionAgent(State):
    """
    Generate plot suggestions from data files using PlotSuggester.
    Integrates file-based workflow with the unified PlotSuggester class.
    """
    
    def __init__(self, model: ChatOpenAI):
        super().__init__(model, "plot_suggestion")
        # Initialize the unified PlotSuggester
        self.plotter = PlotSuggester(model)

    def run(self, state: AgentState) -> dict:
        """
        Generate plot suggestions based on paper plan, content, and data files.
        """
        logging.info(f"state {self.name}: running")
        
        plan = state.plan
        task = state.task
        content = "\n\n".join(state.content or [])
        
        suggested_plots: List[PlotSuggestion] = []
        
        # Process each relevant data file
        for i, relevant_file in enumerate(state.relevant_files):
            if relevant_file.file_path.suffix not in SUPPORTED_TABLE_FORMATS:
                continue
            
            file_path = relevant_file.file_path
            logging.info(f"Processing data file: {file_path}")
            
            try:
                data_preview, columns_desc, shape = extract_table_data(file_path)
                
                # Build context for this file
                section_applications = "\n".join([
                    f"- Section '{app.stage_name}': {app.application_desc}"
                    for app in relevant_file.application
                ])
                
                # Create descriptive prompt for PlotSuggester
                plot_description = self._build_plot_description(
                    file_info={
                        'filename': file_path.name,
                        'columns': columns_desc,
                        'shape': shape,
                        'preview': data_preview,
                        'description': relevant_file.description,
                        'applications': section_applications
                    },
                    paper_context={
                        'plan': plan,
                        'content_preview': content[:2000],
                        'task': task
                    }
                )
                
                # Load the actual data for plot generation
                data_df = self._load_dataframe(file_path)
                
                # Generate multiple plot variations for this file
                num_variations = 3  # Generate 3 variations per file
                
                for variation_idx in range(num_variations):
                    try:
                        # Use PlotSuggester to generate plot
                        fig, code = self.plotter.suggest_plot(
                            paper_content=content,
                            user_prompt=plot_description,
                            num_plots=num_variations
                        )
                        
                        # Create PlotSuggestion object
                        plot_suggestion = PlotSuggestion(
                            id=str(uuid.uuid4()),
                            description=f"Visualization {variation_idx + 1} for {file_path.name}",
                            code=code,
                            rationale=f"Generated from {file_path.name} - {relevant_file.description}",
                            approved=False,
                            filename_base=file_path,
                            figure=fig
                        )
                        
                        suggested_plots.append(plot_suggestion)
                        logging.info(f"✓ Generated plot variation {variation_idx + 1} for {file_path.name}")
                        
                    except Exception as e:
                        logging.error(f"Failed to generate plot variation {variation_idx + 1}: {e}")
                        continue
                
            except Exception as e:
                logging.error(f"Failed to process file {file_path}: {e}")
                continue
        
        # Cleanup matplotlib resources
        self.plotter.cleanup()
        
        # Format summary for display
        # summary = self._format_plot_summary(suggested_plots)
        
        return {
            "state": self.name,
            "suggested_plots": suggested_plots
            # "draft": summary,
        }
    
    def _build_plot_description(self, file_info: dict, paper_context: dict) -> str:
        """
        Build a descriptive prompt for PlotSuggester based on file and paper context.
        """
        description = f"""Create a publication-quality visualization for the paper.

**Data Source:** {file_info['filename']}
- Columns: {file_info['columns']}
- Size: {file_info['shape']}
- Purpose: {file_info['description']}

**Paper Context:**
{paper_context['content_preview']}

**Section Applications:**
{file_info['applications']}

**Data Preview:**
{file_info['preview'][:500]}

Generate an informative, publication-ready plot that:
1. Supports the paper's narrative
2. Highlights key patterns in the data
3. Uses appropriate visualization type
4. Includes clear labels and legend
"""
        return description
    
    def _load_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load data file into DataFrame."""
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_path.suffix == '.tsv':
            return pd.read_csv(file_path, sep='\t')
        elif file_path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    # def _format_plot_summary(self, plots: List[PlotSuggestion]) -> str:
    #     """Format plot suggestions for display."""
    #     if not plots:
    #         return "No plots generated."
        
    #     summary = f"## Plot Suggestions ({len(plots)} generated)\n\n"
        
    #     for i, plot in enumerate(plots, 1):
    #         status = "⏳ Pending Approval" if not plot.approved else "✅ Approved"
            
    #         summary += f"### Plot {i}: {plot.description} [{status}]\n\n"
    #         summary += f"**Source:** {plot.filename_base.name if plot.filename_base else 'N/A'}\n\n"
    #         summary += f"**Rationale:** {plot.rationale}\n\n"
    #         summary += f"**Code Preview:**\n```python\n{plot.code[:300]}...\n```\n\n"
    #         summary += "---\n\n"
        
    #     summary += "\n**Next Step:** Review plots and approve/reject them before generation.\n"
    #     summary += "Type 'approve 1,2,3' or 'reject 2' in the input field.\n"
        
    #     return summary