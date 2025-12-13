import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import scipy
from typing import Tuple, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import matplotlib
import logging
import traceback
from io import StringIO
from src.agents.prompts import PLOT_SUGGESTION_PROMPT

logging.basicConfig(level=logging.DEBUG)

MAX_PLOT_ATTEMPTS = 3
PLOT_DELIMITER = "# ------"


class PlotSuggester:
    """
    Agent that suggests and generates plots based on paper content.
    """

    def __init__(self, model: ChatOpenAI):
        self.model = model
        self._setup_plot_environment()

    def _setup_plot_environment(self):
        """Configure matplotlib for non-interactive backend"""
        matplotlib.use('Agg')
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
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are purely markdown artifacts
            if stripped in ['```', '```python', '```py']:
                continue
            # Skip lines like "Here's the code:" or "Generated Code:"
            if re.match(r'^(Here\'s|Generated|Corrected|Fixed)\s+(the\s+)?code', stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)

        code = '\n'.join(cleaned_lines)

        # Log if we made significant changes
        if len(code) != len(original_code):
            logging.info(f"Code cleaning removed {len(original_code) - len(code)} characters")
            logging.debug(f"Original first 200 chars: {original_code[:200]}")
            logging.debug(f"Cleaned first 200 chars: {code[:200]}")

        return code.strip()

    def _generate_plot_prompt(
        self,
        paper_content: str,
        user_prompt: str,
        error_history: str = "",
        plot_index: int = 1,
        total_plots: int = 1,
    ):
        system_prompt = PLOT_SUGGESTION_PROMPT

        # Task intent
        if paper_content.strip():
            task_context = """
Generate publication-quality plots that support the paper.
Use the user's plot description as guidance.
"""
        else:
            task_context = """
Generate exploratory, publication-quality plots based on the user's description.
You must generate any required data yourself.
"""

        intent_context = f"""
**User Plot Description:**
{user_prompt}
"""

        plot_context = ""
        if total_plots > 1:
            plot_context = f"""
This is plot {plot_index} of {total_plots}.
Make it DISTINCT from the others."""

        # Error correction context
        error_context = ""
        if error_history:
            error_context = f"""
⚠️ **PREVIOUS ATTEMPT FAILED - CORRECTION REQUIRED** ⚠️

{error_history}

**CRITICAL:** Return ONLY the corrected Python code.
Do NOT include:
- Explanations or descriptions
- Markdown code fences (```)
- Comments about what you changed
- Just the raw Python code that will execute successfully.
"""

        user_message = f"""
{task_context}

{intent_context}

**Paper Context:**
{paper_content[:2000] if paper_content.strip() else "(No paper context provided)"}

{plot_context}

{error_context}

**Output Requirements:**
- Return ONLY executable Python code
- NO markdown fences (no ```)
- NO explanations or text before/after
- Must create a figure variable named 'fig'
- Include all necessary imports
- Generate any required data using numpy/pandas

Start with imports and end with the last line of code. Nothing else.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        return messages

    def suggest_plot(
        self,
        paper_content: str = "",
        user_prompt: str = "",
        num_plots: int = 5
    ) -> Tuple[Optional[plt.Figure], str]:
        """
        Generate a publication-quality plot based on paper content and/or user prompt.

        Args:
            paper_content: Optional textual draft to provide context.
            user_prompt: User-provided description of desired plot.
            num_plots: Number of variations (used for prompt context).

        Returns:
            Tuple of (matplotlib Figure, Python code string)
        """

        current_attempt = 1
        error_history = ""
        last_code = ""

        while current_attempt <= MAX_PLOT_ATTEMPTS:
            try:
                # Generate LLM messages
                messages = self._generate_plot_prompt(
                    paper_content=paper_content,
                    user_prompt=user_prompt,
                    error_history=error_history,
                    plot_index=1,
                    total_plots=num_plots
                )

                response = self.model.invoke(messages)
                raw_code = response.content

                code = self._clean_code(raw_code)
                last_code = code

                logging.info(f"Attempt {current_attempt}: Generated {len(code)} chars of code")
                logging.info(f"Code preview:\n{code[:500]}")

                # Close any existing figures
                plt.close('all')

                # Execute the cleaned code
                fig = self._execute_plot_code(code)

                logging.info(f"✓ Plot generated successfully on attempt {current_attempt}")
                return fig, code

            except Exception as e:
                if current_attempt < MAX_PLOT_ATTEMPTS:
                    full_traceback = traceback.format_exc()

                    logging.warning(
                        f"Plot generation attempt {current_attempt}/{MAX_PLOT_ATTEMPTS} failed: {str(e)}"
                    )

                    # Build error history WITHOUT markdown fences
                    # This prevents the LLM from echoing them back
                    error_history = f"""
ATTEMPT {current_attempt} FAILED

The following Python code caused an error:

{last_code}

Error Message: {str(e)}

Error Traceback:
{full_traceback}

Common issues to check:
1. Is 'fig' variable explicitly created?
2. Are all imports included?
3. Are there any undefined variables?
4. Is the data generation logic correct?

Generate CORRECTED code that fixes these issues.
"""

                    current_attempt += 1
                else:
                    logging.error(
                        f"❌ Plot generation failed after {MAX_PLOT_ATTEMPTS} attempts. "
                        f"Last error: {str(e)}"
                    )
                    return self._create_fallback_plot(error_message=str(e))

        return self._create_fallback_plot(error_message="Max attempts reached")

    def _execute_plot_code(
        self,
        code: str,
        data: Optional[pd.DataFrame] = None
    ) -> plt.Figure:
        """
        Execute LLM-generated plotting code in a controlled environment.

        Args:
            code: Python code string (already cleaned of markdown)
            data: Optional DataFrame (not used in current version)

        Returns:
            Generated matplotlib Figure
        """
        namespace = {
            'matplotlib': matplotlib,
            'plt': plt,
            'pd': pd,
            'sns': sns,
            'np': np,
            'scipy': scipy,
            'data': data,
            'PLOT_DELIMITER': PLOT_DELIMITER
        }

        try:
            if '```' in code:
                logging.warning("Found ``` in code during execution, cleaning again...")
                code = self._clean_code(code)

            exec(code, namespace)

            # Try to retrieve figure
            if 'fig' in namespace and isinstance(namespace['fig'], plt.Figure):
                return namespace['fig']
            elif plt.gcf() and len(plt.gcf().axes) > 0:
                return plt.gcf()
            else:
                raise ValueError(
                    "Code executed but did not create a valid Figure. "
                    "Ensure code creates a figure and assigns it to variable 'fig'."
                )

        except Exception as e:
            logging.error(f"Error executing plot code: {e}")
            # Log the problematic code for debugging
            logging.debug(f"Failed code (first 1000 chars):\n{code[:1000]}")
            raise

    def _create_fallback_plot(
        self,
        data: Optional[pd.DataFrame] = None,
        error_message: str = "Plot generation failed"
    ) -> Tuple[plt.Figure, str]:
        """
        Create a fallback error plot when generation fails.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.text(
            0.5, 0.6,
            "⚠️ Plot Generation Failed",
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='#d32f2f'
        )

        ax.text(
            0.5, 0.4,
            error_message[:100],
            ha='center', va='center',
            fontsize=11,
            color='#666'
        )

        if data is not None:
            info_text = f"Data available: {data.shape[0]} rows × {data.shape[1]} cols"
        else:
            info_text = "No data provided"

        ax.text(
            0.5, 0.25,
            info_text,
            ha='center', va='center',
            fontsize=9,
            color='#999'
        )

        ax.text(
            0.5, 0.15,
            "Try regenerating or check your data format",
            ha='center', va='center',
            fontsize=9,
            style='italic',
            color='#999'
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

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
        plt.close('all')