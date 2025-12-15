import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.utils.codeapi import CodeAPI
from src.utils.models import RelevantFile

matplotlib.use("Agg")
plt.ioff()


class SimplePlotGenerator:
    """
    Streamlined plot generator with minimal overhead.
    Focus: Generate clean, focused plotting code.
    """
    
    SYSTEM_PROMPT = """You are a data visualization expert. Generate clean, minimal Python code for plots.

CRITICAL RULES:
1. Use ONLY necessary imports
2. Code must be 15-30 lines maximum
3. Read data from 'data.csv' file
4. Create ONE figure with plt.subplots()
5. Save to 'plot.png' at the end
6. NO explanations, NO comments, NO markdown
7. Focus on clarity over complexity

Example structure:
```
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
fig, ax = plt.subplots(figsize=(10, 6))
# ... your plot code ...
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```"""

    def __init__(self, model: ChatOpenAI, working_dir: Path):
        self.model = model
        self.codeapi = CodeAPI()
        self.working_dir = working_dir
        self.max_attempts = 3

    def generate_plot(
        self,
        relevant_file: RelevantFile,
        user_prompt: str,
        paper_context: str = "",
    ) -> Tuple[Optional[Path], str]:
        """
        Generate a single plot with minimal overhead.
        
        Args:
            relevant_file: Data file to plot
            user_prompt: What the user wants to visualize
            paper_context: Optional paper context (kept short)
            
        Returns:
            (image_path, code) or (None, error_code)
        """
        
        # Build focused prompt
        prompt = self._build_prompt(relevant_file, user_prompt, paper_context)
        
        # Try generation with retries
        for attempt in range(1, self.max_attempts + 1):
            try:
                logging.info(f"Plot generation attempt {attempt}/{self.max_attempts}")
                
                # Get code from LLM
                response = self.model.invoke([
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=prompt)
                ])
                
                code = self._clean_code(response.content)
                
                # Execute and save
                img_path = self._execute_and_save(code, relevant_file)
                
                logging.info(f"✓ Plot generated successfully")
                return img_path, code
                
            except Exception as e:
                logging.warning(f"Attempt {attempt} failed: {e}")
                
                if attempt < self.max_attempts:
                    # Add error feedback to prompt for next attempt
                    prompt = self._add_error_context(prompt, str(e), code)
                else:
                    # Final attempt failed
                    logging.error(f"All {self.max_attempts} attempts failed")
                    return None, f"# Generation failed after {self.max_attempts} attempts\n# Error: {e}"
        
        return None, "# Unknown error"

    def _build_prompt(
        self,
        relevant_file: RelevantFile,
        user_prompt: str,
        paper_context: str
    ) -> str:
        """Build a focused, concise prompt."""
        
        # Get basic file info
        file_desc = relevant_file.description or "Data file for analysis"
        
        # Keep paper context SHORT (max 500 chars)
        context_snippet = paper_context[:500] + "..." if len(paper_context) > 500 else paper_context
        
        prompt = f"""Generate a plot for this data visualization request:

**Request:** {user_prompt}

**Data File:** {relevant_file.file_path.name}
**Purpose:** {file_desc}

**Paper Context:** {context_snippet if context_snippet else "General exploratory analysis"}

**Requirements:**
- Read from 'data.csv' 
- Create ONE clear, publication-quality plot
- Use appropriate plot type for the data
- Include axis labels and title
- Keep code under 30 lines
- Return ONLY the Python code, no explanations

Generate the code now:"""
        
        return prompt

    def _clean_code(self, raw_code: str) -> str:
        """Remove markdown fences and extract pure Python code."""
        
        # Remove markdown code blocks
        code = re.sub(r'```python\s*\n?', '', raw_code)
        code = re.sub(r'```\s*$', '', code)
        code = re.sub(r'^```\s*\n?', '', code)
        
        # Remove any remaining backticks
        code = code.replace('```', '')
        
        # Remove empty lines at start/end
        code = code.strip()
        
        # Validate basic structure
        if 'import' not in code or 'plt' not in code:
            raise ValueError("Generated code missing required imports")
        
        return code

    def _execute_and_save(
        self,
        code: str,
        relevant_file: RelevantFile
    ) -> Path:
        """Execute code and save the generated plot."""
        
        # Read data file
        with relevant_file.file_path.open('r', encoding='utf-8') as f:
            file_data = f.read()
        
        # Execute code via CodeAPI
        result = self.codeapi.run_python(code, {'data.csv': file_data})
        
        # Extract images
        images_b64 = CodeAPI.extract_images_base64(result)
        
        if not images_b64:
            raise ValueError("No plot image was generated")
        
        # Save first image
        images_dir = self.working_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate simple filename
        filename = f"{relevant_file.file_path.stem}_plot.png"
        img_path = images_dir / filename
        
        # Save image
        img_bytes = CodeAPI.b64_to_bytes(images_b64[0])
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
        
        logging.info(f"Saved plot: {img_path}")
        return img_path

    def _add_error_context(
        self,
        original_prompt: str,
        error: str,
        failed_code: str
    ) -> str:
        """Add error feedback for retry attempt."""
        
        return f"""{original_prompt}

**PREVIOUS ATTEMPT FAILED - FIX THE ISSUES:**

Error: {error}

Failed code:
{failed_code}

Generate CORRECTED code that fixes this error. Focus on:
1. Correct column names
2. Proper data types
3. Valid plot syntax
4. No undefined variables

Return ONLY the corrected Python code:"""

    def cleanup(self):
        """Clean up matplotlib resources."""
        plt.close('all')

