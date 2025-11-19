import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import matplotlib
import logging
import traceback 
from io import StringIO 
from src.agents.prompts import PLOT_SUGGESTION_PROMPT
from pathlib import Path
import os
from io import BytesIO
from src.utils.codeapi import CodeAPI
import time
import json

logging.basicConfig(level=logging.WARNING)

MAX_PLOT_ATTEMPTS = 3
PLOT_DELIMITER = "------"
CAPTURE_FIGS_TO_STDOUT = r"""
try:
    import matplotlib.pyplot as _plt
    import io as _io, base64 as _b64
    figs = _plt.get_fignums()
    if not figs:
        _plt.figure()
        figs = _plt.get_fignums()
    for _i in figs:
        _fig = _plt.figure(_i)
        _buf = _io.BytesIO()
        _fig.savefig(_buf, format='png', bbox_inches='tight')
        _buf.seek(0)
        _enc = _b64.b64encode(_buf.getvalue()).decode('ascii')
        print('data:image/png;base64,' + _enc)
except Exception as _e:
    print('capture_error:', _e)
"""

class PlotSuggester:
    """
    Agent that suggests and generates plots based on paper content.
    """

    def __init__(self, model: ChatOpenAI):
        self.model = model
        self._codeapi_url = os.environ.get("CODEAPI_URL")
        self._codeapi: Optional[CodeAPI] = (
            CodeAPI(self._codeapi_url) if self._codeapi_url else None
        )

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
        num_plots: int = 3,
        save_dir: Optional[Path] = None,
        use_remote: bool = False,
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

                if use_remote and self._codeapi is not None:
                    fig = self._run_code_remote(code, data, save_dir)
                    if fig is not None:
                        return fig, code
                    raise RuntimeError("Remote execution returned no images")

                fig = self._execute_plot_code(code, data)
                return fig, code

            except Exception as e:
                if current_attempt < MAX_PLOT_ATTEMPTS and not use_remote:
                    full_traceback = traceback.format_exc()
                    logging.warning(f"Plot generation attempt {current_attempt} failed.")
                    error_history += f"\n!!!Attempt {current_attempt} Failed\n"
                    error_history += f"Code (first 300 chars):\n{last_code[:300]}...\n"
                    error_history += f"Execution Error Traceback:\n{full_traceback}"
                    current_attempt += 1
                else:
                    logging.error(
                        f"Plot code generation/execution failed. use_remote={use_remote}. Last Error: {e}"
                    )
                    return None, last_code

        return self._create_fallback_plot(data)

    def _run_code_remote(
        self,
        code: str,
        data: Optional[pd.DataFrame],
        save_dir: Optional[Path],
    ) -> Optional[plt.Figure]:
        try:
            if data is not None:
                csv_data = data.to_csv(index=False)
                bootstrap = (
                    "import pandas as pd\n"
                    "from io import StringIO\n"
                    f"_CSV='''{csv_data}'''\n"
                    "data = pd.read_csv(StringIO(_CSV))\n"
                )
                code_to_run = bootstrap + "\n" + code
            else:
                code_to_run = code

            code_to_run = code_to_run + "\n" + CAPTURE_FIGS_TO_STDOUT

            result = self._codeapi.run_python(code_to_run)

            images_b64 = CodeAPI.extract_images_base64(result)
            logging.warning(f"CodeAPI extracted {len(images_b64)} image(s)")
            if not images_b64:
                logging.warning("CodeAPI returned no images; falling back to local execution")
                return None

            first_img_b64 = images_b64[0]
            img_bytes = CodeAPI.b64_to_bytes(first_img_b64)

            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                for idx, b64 in enumerate(images_b64, start=1):
                    b = CodeAPI.b64_to_bytes(b64)
                    out_path = save_dir / f"plot_{int(time.time())}_{idx}.png"
                    with open(out_path, "wb") as f:
                        f.write(b)

            from PIL import Image as PILImage
            import numpy as _np
            buf = BytesIO(img_bytes)
            pil_img = PILImage.open(buf).convert("RGBA")
            arr = _np.array(pil_img)
            fig, ax = plt.subplots()
            ax.imshow(arr)
            ax.axis("off")
            fig.tight_layout()
            return fig
        except Exception as e:
            logging.error(f"Remote execution via CodeAPI failed: {e}")
            return None

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
                'print': print,
                'dict': dict, 'list': list, 'set': set, 'tuple': tuple,
                'int': int, 'float': float, 'str': str, 'bool': bool,
                'len': len, 'range': range, 'type': type, 'isinstance': isinstance,
                'getattr': getattr, 'hasattr': hasattr,
                'min': min, 'max': max, 'round': round, 'abs': abs, 'sum': sum,
                'enumerate': enumerate, 'zip': zip, 'sorted': sorted, 'any': any, 'all': all,
                '__import__': __import__,
            }
        }

        try:
            exec(code, namespace, namespace)
            
            if 'fig' in namespace and isinstance(namespace['fig'], plt.Figure):
                return namespace['fig']
            elif plt.gcf():
                return plt.gcf()
            else:
                raise ValueError("The generated code did not explicitly create or return a `matplotlib.figure.Figure` object named 'fig'.")

        except Exception as e:
            logging.error(f"Error executing plot code: {e}")
            raise

    def _create_fallback_plot(self, data: Optional[pd.DataFrame] = None) -> Tuple[plt.Figure, str]:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Plot generation unavailable", ha='center', va='center')
        ax.axis('off')
        return fig, ""

    def run_code_remote_save(
        self,
        code: str,
        data: Optional[pd.DataFrame],
        save_dir: Path,
    ) -> List[Path]:
        """
        Execute given plotting code on real data via CodeAPI and save images to disk.
        Returns list of saved image paths. Raises if CODEAPI is not configured.
        """
        if self._codeapi is None:
            raise RuntimeError("CODEAPI_URL is not configured")
        if data is not None:
            csv_data = data.to_csv(index=False)
            bootstrap = (
                "import pandas as pd\n"
                "from io import StringIO\n"
                f"_CSV='''{csv_data}'''\n"
                "data = pd.read_csv(StringIO(_CSV))\n"
            )
            code_to_run = bootstrap + "\n" + code
        else:
            code_to_run = code
        code_to_run = code_to_run + "\n" + CAPTURE_FIGS_TO_STDOUT

        result = self._codeapi.run_python(code_to_run)

        images_b64 = CodeAPI.extract_images_base64(result)
        logging.warning(f"CodeAPI extracted {len(images_b64)} image(s)")
        saved: List[Path] = []
        if not images_b64:
            return saved
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        for idx, b64 in enumerate(images_b64, start=1):
            b = CodeAPI.b64_to_bytes(b64)
            out_path = save_dir / f"plot_{ts}_{idx}.png"
            with open(out_path, "wb") as f:
                f.write(b)
            saved.append(out_path)
        return saved
