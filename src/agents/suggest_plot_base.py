import logging
import re
import traceback
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.prompts import PLOT_SUGGESTION_PROMPT
from src.agents.utils import create_document_prompt
from src.utils.codeapi import CodeAPI
from src.utils.constants import CAPTURE_CODE, DATA_BOOTSTRAP, MATPLOTLIB_SETUP
from src.utils.models import RelevantFile

MAX_PLOT_ATTEMPTS = 3
PLOT_DELIMITER = "# ------"


class PlotSuggester:
    """
    Agent that suggests and generates plots based on paper content.
    """

    def __init__(self, model: ChatOpenAI, working_dir: Path):
        self.model = model
        self.codeapi = CodeAPI()
        self.working_dir = working_dir
        self._setup_plot_environment()

    def _setup_plot_environment(self):
        """Configure matplotlib for non-interactive backend"""
        matplotlib.use("Agg")
        plt.ioff()

    def _extract_data_summary(self, data: pd.DataFrame, max_rows: int = 5) -> str:
        """
        Extract comprehensive data summary for LLM context.

        Args:
            data: DataFrame to summarize
            max_rows: Number of sample rows to include

        Returns:
            Formatted string with data information
        """
        try:
            buffer = StringIO()
            data.info(buf=buffer)
            info_str = buffer.getvalue()

            summary = f"""
Data Schema:
{info_str}

Columns: {data.columns.tolist()}
Shape: {data.shape[0]} rows × {data.shape[1]} columns

Sample Data (first {max_rows} rows):
{data.head(max_rows).to_string()}

Statistical Summary (numeric columns only):
{data.describe().to_string()}
"""
            return summary
        except Exception as e:
            logging.error(f"Error extracting data summary: {e}")
            return f"Columns: {data.columns.tolist()}\nShape: {data.shape}"

    def _clean_code(self, code: str) -> str:
        """
        Aggressively clean code to remove ALL markdown artifacts.

        This handles cases where:
        1. Code starts/ends with ```python or ```
        2. Code has markdown fences in the middle (from error history)
        3. LLM echoes back the error history with fences

        Args:
            code: Raw code string from LLM

        Returns:
            Clean Python code without any markdown
        """
        original_code = code

        # Step 1: Remove markdown code fences from START and END
        code = re.sub(r"^```(?:python|py)?\s*\n?", "", code.strip())
        code = re.sub(r"\n?```\s*$", "", code.strip())

        # Step 2: Remove any remaining ``` markers in the middle
        # This catches cases where error history gets echoed back
        code = re.sub(r"```(?:python|py)?\s*\n?", "", code)
        code = re.sub(r"\n?```", "", code)

        # Step 3: Remove common markdown artifacts
        # Remove lines that are just markdown or comments about code blocks
        lines = code.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are purely markdown artifacts
            if stripped in ["```", "```python", "```py"]:
                continue
            # Skip lines like "Here's the code:" or "Generated Code:"
            if re.match(
                r"^(Here\'s|Generated|Corrected|Fixed)\s+(the\s+)?code",
                stripped,
                re.IGNORECASE,
            ):
                continue
            cleaned_lines.append(line)

        code = "\n".join(cleaned_lines)

        # Log if we made significant changes
        if len(code) != len(original_code):
            logging.info(
                f"Code cleaning removed {len(original_code) - len(code)} characters"
            )
            logging.debug(f"Original first 200 chars: {original_code[:200]}")
            logging.debug(f"Cleaned first 200 chars: {code[:200]}")

        draft_code = code.strip()

        # return f"{MATPLOTLIB_SETUP}\n{DATA_BOOTSTRAP}\n{draft_code}\n{CAPTURE_CODE}"
        return f"{MATPLOTLIB_SETUP}\n{draft_code}\n{CAPTURE_CODE}"

    def _generate_plot_prompt(
        self,
        relevant_file: RelevantFile,
        paper_content: str,
        user_prompt: str,
        error_history: str = "",
    ):
        # --- Core task ---
        user_message = f"""
User request:
{user_prompt}

Data context:
The data comes from a file with the following structure:
{create_document_prompt(relevant_file)}
    """

        # --- Optional paper context (SHORT) ---
        if paper_content.strip():
            user_message += f"""

Paper context (high-level, optional):
{paper_content[:800]}
    """

        # --- Retry correction ONLY when needed ---
        if error_history:
            user_message += f"""

The previous code failed with this error:

{error_history}

Return corrected Python code that fixes the error.
"""

        return [
            SystemMessage(content=PLOT_SUGGESTION_PROMPT),
            HumanMessage(content=user_message.strip()),
        ]


    def suggest_plot(
        self,
        relevant_file: RelevantFile,
        paper_content: str = "",
        user_prompt: str = ""
    ) -> Tuple[Optional[Path], str]:
        """
        Generate a publication-quality plot based on paper content and/or user prompt.

        Args:
            paper_content: Optional textual draft to provide context.
            user_prompt: User-provided description of desired plot.
            num_plots: Number of variations (used for prompt context).
            data_df: Optional preloaded DataFrame to avoid disk I/O.

        Returns:
            Tuple of (image path, Python code string)
        """

        current_attempt = 1
        error_history = ""

        while current_attempt <= MAX_PLOT_ATTEMPTS:
            try:
                # Generate LLM messages
                messages = self._generate_plot_prompt(
                    relevant_file=relevant_file,
                    paper_content=paper_content,
                    user_prompt=user_prompt,
                    error_history=error_history,
                    # plot_index=1,
                    # total_plots=num_plots,
                )

                response = self.model.invoke(messages)
                raw_code = response.content

                code = self._clean_code(raw_code)
                last_code = code

                logging.info(
                    f"Attempt {current_attempt}: Generated {len(code)} chars of code"
                )
                logging.info(f"Code preview:\n{code[:500]}")

                result = self.codeapi.run_python(code, {})

                # Prepare files payload: use cached DataFrame if provided, else read from disk
                # TODO: think if we need tha part:
                # if data_df is not None:
                #     try:
                #         file_data = data_df.to_csv(index=False)
                #         payload_name = "data.csv"
                #     except Exception as e:
                #         logging.warning(f"Failed to convert DataFrame to CSV, falling back to disk read: {e}")
                #         with relevant_file.file_path.open("r", encoding="utf-8") as f:
                #             file_data = f.read()
                #         payload_name = "data.csv"
                # else:
                #     with relevant_file.file_path.open("r", encoding="utf-8") as f:
                #         file_data = f.read()
                #     payload_name = "data.csv"

                # result = self.codeapi.run_python(code, {payload_name: file_data})


                images_b64 = CodeAPI.extract_images_base64(result)

                img_path: Optional[Path] = None
                if images_b64:
                    filename_base = self.generate_plot_name(code)

                    images_dir = self.working_dir / "images"
                    images_dir.mkdir(parents=True, exist_ok=True)

                    for j, b64_img in enumerate(images_b64):
                        img_bytes = CodeAPI.b64_to_bytes(b64_img)

                        if len(images_b64) > 1:
                            filename = f"{filename_base}_{j + 1}.png"
                        else:
                            filename = f"{filename_base}.png"

                        img_path = images_dir / filename

                        with open(img_path, "wb") as f:
                            f.write(img_bytes)

                        # Store both the full path and the relative path for markdown
                        relative_path = f"images/{filename}"

                        logging.info(f"Saved plot image: {img_path}")
                else:
                    logging.warning("No images generated for plot")
                    # Trigger retry/fallback path
                    raise ValueError("No images generated from code execution")

                logging.info(
                    f"✓ Plot generated successfully on attempt {current_attempt}"
                )
                return img_path, code

            except Exception as e:
                if current_attempt < MAX_PLOT_ATTEMPTS:
                    full_traceback = traceback.format_exc()

                    logging.warning(
                        f"Plot generation attempt {current_attempt}/{MAX_PLOT_ATTEMPTS} failed: {str(e)}"
                    )

                    error_history = f"""
Python error:
{str(e)}

Traceback:
{traceback.format_exc()}
"""

                    current_attempt += 1
                else:
                    logging.error(
                        f"❌ Plot generation failed after {MAX_PLOT_ATTEMPTS} attempts. "
                        f"Last error: {str(e)}"
                    )
                    return self._create_fallback_plot(error_message=str(e))

        return self._create_fallback_plot(error_message="Max attempts reached")

    # def _execute_plot_code(
    #     self, code: str, data: Optional[pd.DataFrame] = None
    # ) -> plt.Figure:
    #     """
    #     Execute LLM-generated plotting code in a controlled environment.

    #     Args:
    #         code: Python code string (already cleaned of markdown)
    #         data: Optional DataFrame (not used in current version)

    #     Returns:
    #         Generated matplotlib Figure
    #     """
    #     namespace = {
    #         "matplotlib": matplotlib,
    #         "plt": plt,
    #         "pd": pd,
    #         "sns": sns,
    #         "np": np,
    #         "scipy": scipy,
    #         "data": data,
    #         "PLOT_DELIMITER": PLOT_DELIMITER,
    #     }

    #     try:
    #         if "```" in code:
    #             logging.warning("Found ``` in code during execution, cleaning again...")
    #             code = self._clean_code(code)

    #         self.codeapi.run_python(code)

    #         exec(code, namespace)

    #         # Try to retrieve figure
    #         if "fig" in namespace and isinstance(namespace["fig"], plt.Figure):
    #             return namespace["fig"]
    #         elif plt.gcf() and len(plt.gcf().axes) > 0:
    #             return plt.gcf()
    #         else:
    #             raise ValueError(
    #                 "Code executed but did not create a valid Figure. "
    #                 "Ensure code creates a figure and assigns it to variable 'fig'."
    #             )

    #     except Exception as e:
    #         logging.error(f"Error executing plot code: {e}")
    #         # Log the problematic code for debugging
    #         logging.debug(f"Failed code (first 1000 chars):\n{code[:1000]}")
    #         raise

    def generate_plot_name(self, code: str) -> str:
        result = self.model.invoke(
            [
                SystemMessage(
                    content=f"You will be given a Python code that generates a plot. You have to describe it with a short and consice name. You must output name in max 5 words separated by underscores. You MUST NOT include any other text other than the name."
                ),
                HumanMessage(
                    content=f"Here is the code: {code}. Create a name that describes the plot generated by this code."
                ),
            ]
        )

        return result.content.strip()

    def _create_fallback_plot(
        self,
        data: Optional[pd.DataFrame] = None,
        error_message: str = "Plot generation failed",
    ) -> Tuple[plt.Figure, str]:
        """
        Create a fallback error plot when generation fails.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.text(
            0.5,
            0.6,
            "⚠️ Plot Generation Failed",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#d32f2f",
        )

        ax.text(
            0.5,
            0.4,
            error_message[:100],
            ha="center",
            va="center",
            fontsize=11,
            color="#666",
        )

        if data is not None:
            info_text = f"Data available: {data.shape[0]} rows × {data.shape[1]} cols"
        else:
            info_text = "No data provided"

        ax.text(
            0.5, 0.25, info_text, ha="center", va="center", fontsize=9, color="#999"
        )

        ax.text(
            0.5,
            0.15,
            "Try regenerating or check your data format",
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            color="#999",
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        fallback_code = f"""# Fallback plot - Generation failed
# Error: {error_message}

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.text(0.5, 0.5, 'Plot generation failed',
        ha='center', va='center', fontsize=14)
ax.axis('off')
"""
        return fig, fallback_code

    def cleanup(self):
        """Clean up matplotlib resources"""
        plt.close("all")
