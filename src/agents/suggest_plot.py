import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import matplotlib
import logging
import traceback 
from io import StringIO 
from prompts import PLOT_SUGGESTION_PROMPT
logging.basicConfig(level=logging.WARNING)

MAX_PLOT_ATTEMPTS = 3
PLOT_DELIMITER = "------"
class PlotSuggester:
    """
    Agent that suggests and generates plots based on paper content.
    """

    def __init__(self, model: ChatOpenAI):
        self.model = model

    def _generate_plot_prompt(
        self, 
        paper_content: str, 
        data: Optional[pd.DataFrame] = None,
        error_history: str ='',
    ) -> Tuple[Optional[plt.Figure], str]:
        try:
            system_prompt = PLOT_SUGGESTION_PROMPT
            data_context = ""
            if data is not None:
                data_info = data.info(buf=StringIO())
                data_schema = StringIO(data_info).getvalue()
                data_context = f"\n\nData Context (Available as `data` DataFrame):\n"
                data_context += f"The data schema is:\n{data_schema}"
                data_context += f"\nData Columns: {data.columns.tolist()}"
            else:
                data_context = "\n\nNo example data is provided. Generate appropriate sample data that fits the likely topic (e.g., time series, category comparison, correlation)."

            user_prompt = f"""Based on this paper content, suggest and generate the plot:
                {paper_content[:2000]}
                {data_context}
                """

            if error_history:
                user_prompt += f"""
                CRITICAL CORRECTION REQUIRED

                The previous code attempt failed to execute due to an error.
                Please review the error traceback and your previous code history below, and generate the corrected, executable Python code block.
                ERROR HISTORY:{error_history}
                """

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            return messages

        except Exception as e:
            logging.error(f"Error in suggest_plot: {e}")
            return

    def suggest_plot(
        self, 
        paper_content: str, 
        data: Optional[pd.DataFrame] = None,
        num_plots: int = 3
    ) -> Tuple[Optional[plt.Figure], str]:
        current_attempt = 1
        error_history = ""
        last_code = ""

        while current_attempt <= MAX_PLOT_ATTEMPTS:
            try:
                messages = self._generate_plot_prompt(paper_content, data, error_history)

                response = self.model.invoke(messages)
                code = response.content
                last_code = code

                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0].strip()
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0].strip()

                fig = self._execute_plot_code(code, data)
                
                return fig, code

            except Exception as e:
                if current_attempt < MAX_PLOT_ATTEMPTS:
                    full_traceback = traceback.format_exc()
                    
                    logging.warning(f"Plot generation attempt {current_attempt} failed.")
                    
                    error_history += f"\n!!!Attempt {current_attempt} Failed\n"
                    error_history += f"Code (first 300 chars):\n{last_code[:300]}...\n"
                    error_history += f"Execution Error Traceback:\n{full_traceback}"
                    
                    current_attempt += 1
                else:
                    logging.error(f"LLM failed to generate executable code after {MAX_PLOT_ATTEMPTS} attempts.")
                    logging.error(f"# LLM Plot Generation Failed after {MAX_PLOT_ATTEMPTS} Retries. Last Error: {e}")
                    return fig, code
        
        return self._create_fallback_plot(data)

    def _execute_plot_code(
        self, 
        code: str, 
        data: Optional[pd.DataFrame] = None
    ) -> plt.Figure:
        namespace = {
            'matplotlib': matplotlib,
            'plt': plt,
            'pd': pd,
            'sns': sns,
            'np': np,
            'data': data,
            '__builtins__': {
                'print': print, 'len': len, 'range': range, 'type': type, 
                'min': min, 'max': max, 'round': round, 'abs': abs
            }
        }

        try:
            exec(code, {}, namespace) 
            
            if 'fig' in namespace and isinstance(namespace['fig'], plt.Figure):
                return namespace['fig']
            elif plt.gcf():
                return plt.gcf()
            else:
                raise ValueError("The generated code did not explicitly create or return a `matplotlib.figure.Figure` object named 'fig'.")

        except Exception as e:
            logging.error(f"Error executing plot code: {e}")
            raise
