# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import logging
import os
import re
import subprocess
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gradio as gr
import markdown
import matplotlib.pyplot as plt
import polars as pl
import yaml
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.generate_plot import *
from src.agents.states import *
from src.agents.suggest_plot import *
from src.agents.suggest_plot import PlotSuggester
from src.kiroku_app import DocumentWriter
from src.utils.models import PaperConfig

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from decouple import config

from src.utils.constants import SUPPORTED_TABLE_FORMATS

NUM_PLOTS = 5
PREINITIALIZED_COMPONENTS = 20  # For file preview slots

@dataclass
class PlotVersion:
    """Represents a generated plot version"""
    id: str
    figure: plt.Figure
    code: str
    timestamp: datetime
    selected: bool = False

    def __hash__(self):
        return hash(self.id)

class SimplifiedKirokuUI:
    """Simplified UI with radio button plot selection"""
    
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.state_values = None
        self.plotter = self._initialize_plotter()
        
        # Plot gallery
        self.plot_gallery = []
        self.selected_plot_id = None  # ← Changed from set to single ID
    
    def _initialize_plotter(self):
        """Initialize PlotSuggester"""
        try:
            model = ChatOpenAI(
                model=config("OPENAI_MODEL_NAME", default="gpt-5-nano"),
                temperature=1.0,
                api_key=config("OPENAI_API_KEY"),
            )
            plotter = PlotSuggester(model)
            logging.info("✓ PlotSuggester initialized")
            return plotter
        except Exception as e:
            logging.error(f"Failed to initialize PlotSuggester: {e}")
            return None
    
    def read_initial_state(self, filename: Path):
        """Read YAML config - keep this from original"""
        with filename.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        relevant_files = self._get_relevant_files(data)
        cfg = PaperConfig(**data, relevant_files=relevant_files)

        return cfg
    
    def _get_relevant_files(self, state_values: dict) -> list:
        """Discover relevant files - keep this from original"""
        working_dir = Path(state_values["working_dir"])

        if not working_dir.exists() or not working_dir.is_dir():
            return []

        file_names_and_descs = {
            rel_file["file_name"]: rel_file.get("description", "")
            for rel_file in state_values.get("files_descriptions", [])
        }

        relevant_files = []
        for file in working_dir.rglob("*"):
            for file_name, description in file_names_and_descs.items():
                if (file.suffix in [".yaml", ".yml", ".md", ".tex", ".html", *SUPPORTED_TABLE_FORMATS]
                    and file.name == file_name):
                    relevant_files.append(
                        RelevantFile(file_path=file, description=description)
                    )

        return relevant_files
    
    def process_file(self, filename: str):
        """Process uploaded YAML"""
        pwd = Path.cwd()
        self.filename = pwd / Path(filename).name
        self.state_values = self.read_initial_state(Path(filename))
        self.working_dir = self.state_values.working_dir

        relevant_files_preview = self._update_files_preview()

        return (
            *relevant_files_preview,
            self.state_values.model_dump_json(),
        )

    def generate_multiple_plots(self, plot_prompt: str, num_plots: int):
        """Generate multiple plot variations"""
        try:
            # Clear previous gallery
            self.plot_gallery = []
            self.selected_plot_id = None

            # Get paper context if available
            paper_context = ""
            if self.state_values and hasattr(self.state_values, 'hypothesis'):
                paper_context = self.state_values.hypothesis

            has_prompt = bool(plot_prompt.strip())
            has_draft = bool(paper_context.strip())

            if not has_prompt and not has_draft:
                return self._create_error_message(
                    "⚠️ No input provided\n\nPlease provide a plot description."
                )

            # Generate plots with VARIATION in the prompt
            for i in range(num_plots):
                logging.info(f"Generating variation {i+1}/{num_plots}...")

                try:
                    # Add variation instruction to prompt
                    varied_prompt = plot_prompt
                    if num_plots > 1:
                        varied_prompt = f"""{plot_prompt}

    **IMPORTANT: This is variation {i+1} of {num_plots}.**
    Make this plot DIFFERENT from other variations by:
    - Using different chart types (line, bar, scatter, heatmap, etc.)
    - Showing different aspects of the data
    - Using different color schemes
    - Highlighting different patterns

    Be creative and distinct!"""

                    fig, code = self.plotter.suggest_plot(
                        paper_content=paper_context if has_draft else "",
                        user_prompt=varied_prompt,  # ← Use varied prompt
                        num_plots=num_plots
                    )

                    plot_version = PlotVersion(
                        id=str(uuid.uuid4()),
                        figure=fig,
                        code=code,  # ← Each should now have different code
                        timestamp=datetime.now(),
                        selected=False
                    )

                    self.plot_gallery.append(plot_version)
                    logging.info(f"✓ Variation {i+1} generated")

                    # Log code preview to verify it's different
                    logging.debug(f"Variation {i+1} code preview: {code[:100]}...")

                except Exception as e:
                    logging.error(f"Failed to generate variation {i+1}: {e}")
                    continue

            success_count = len(self.plot_gallery)
            logging.info(f"✓ Complete: {success_count}/{num_plots} successful")

            return self.render_gallery()

        except Exception as e:
            logging.error("Critical error in generate_multiple_plots", exc_info=True)
            return self._create_error_message(f"Critical Error:\n{str(e)}")
        finally:
            if self.plotter:
                self.plotter.cleanup()

    def render_gallery(self):
        """Render the plot gallery with radio choices"""
        if not self.plot_gallery:
            empty_updates = [gr.update(visible=False)] * NUM_PLOTS
            return (
                *empty_updates,
                gr.update(choices=[], value=None),  # Radio button
                "# No plots generated",
                "Ready"
            )

        # Update plot displays
        plot_updates = []
        for i in range(NUM_PLOTS):
            if i < len(self.plot_gallery):
                plot = self.plot_gallery[i]
                plot_updates.append(gr.update(value=plot.figure, visible=True))
            else:
                plot_updates.append(gr.update(visible=False))

        # Update radio button choices
        radio_choices = [f"Plot {i+1}" for i in range(len(self.plot_gallery))]
        radio_update = gr.update(
            choices=radio_choices,
            value=radio_choices[0] if radio_choices else None,
            visible=True
        )

        # Show first plot's code by default
        code = self.plot_gallery[0].code if self.plot_gallery else "# No code"
        status = f"✓ {len(self.plot_gallery)} plots generated"

        return (*plot_updates, radio_update, code, status)

    def show_single_plot_code(self, plot_choice: str):
        """Show code for selected plot from radio button"""
        if not plot_choice or not self.plot_gallery:
            return "# No plot selected"

        try:
            # Parse "Plot 1", "Plot 2", etc.
            plot_idx = int(plot_choice.split()[-1]) - 1

            if 0 <= plot_idx < len(self.plot_gallery):
                self.selected_plot_id = self.plot_gallery[plot_idx].id
                code = self.plot_gallery[plot_idx].code
                status = f"✓ {len(self.plot_gallery)} plots | Viewing: Plot {plot_idx + 1}"
                return code, status
        except Exception as e:
            logging.error(f"Error showing plot code: {e}")

        return "# Error loading plot code", "Error"

    def _create_error_message(self, message: str):
        """Create error display"""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=11)
        ax.axis('off')

        error_updates = []
        for i in range(NUM_PLOTS):
            if i == 0:
                error_updates.append(gr.update(value=fig, visible=True))
            else:
                error_updates.append(gr.update(visible=False))

        return (
            *error_updates,
            gr.update(choices=[], value=None),  # Radio
            f"# Error\n{message}",
            f"{message}"
        )

    def _make_files_preview(self, n_slots: int = 20):
        """Create file preview components - keep from original"""
        file_blocks = {
            "yaml": [], "markdown": [], "html": [], "table": [], "other": []
        }

        with gr.Column() as col:
            for i in range(n_slots):
                file_blocks["yaml"].append(
                    gr.Code(visible=False, label=f"yaml_{i}", language="yaml")
                )
                file_blocks["markdown"].append(
                    gr.Markdown(visible=False, label=f"md_{i}")
                )
                file_blocks["html"].append(
                    gr.HTML(visible=False, label=f"html_{i}")
                )
                file_blocks["table"].append(
                    gr.Dataframe(visible=False, label=f"table_{i}")
                )
                file_blocks["other"].append(
                    gr.Markdown(visible=False, label=f"other_{i}")
                )

        all_components = [c for lst in file_blocks.values() for c in lst]
        return col, file_blocks, all_components
    
    def _update_files_preview(self) -> list:
        """Update file preview with data - keep from original"""
        relevant_files = self.state_values.relevant_files

        def pad(updates_list: list, n_left: int) -> list:
            return updates_list + [gr.update(visible=False) for _ in range(n_left - len(updates_list))]

        yaml_updates, md_updates, html_updates, table_updates, other_updates = [], [], [], [], []

        for rel_file in relevant_files:
            file = rel_file.file_path
            try:
                if file.suffix in [".yaml", ".yml"]:
                    yaml_updates.append(
                        gr.update(value=file.read_text(), visible=True, label=file.name)
                    )
                elif file.suffix in [".md", ".tex"]:
                    md_updates.append(
                        gr.update(value=file.read_text(), visible=True, label=file.name)
                    )
                elif file.suffix == ".html":
                    html_updates.append(
                        gr.update(value=file.read_text(), visible=True, label=file.name)
                    )
                elif file.suffix == ".csv":
                    table = pl.scan_csv(file).head(5).collect().to_pandas()
                    table_updates.append(
                        gr.update(value=table, visible=True, label=file.name)
                    )
                elif file.suffix == ".parquet":
                    table = pl.scan_parquet(file).head(5).collect().to_pandas()
                    table_updates.append(
                        gr.update(value=table, visible=True, label=file.name)
                    )
                else:
                    other_updates.append(
                        gr.update(value=f"Unsupported: {file}", visible=True, label=file.name)
                    )
            except Exception as e:
                logging.error(f"Error loading {file}: {e}")
                other_updates.append(
                    gr.update(value=f"Error: {file.name}", visible=True, label=file.name)
                )

        updates = []
        updates.extend(pad(yaml_updates, 20))
        updates.extend(pad(md_updates, 20))
        updates.extend(pad(html_updates, 20))
        updates.extend(pad(table_updates, 20))
        updates.extend(pad(other_updates, 20))

        return updates
    
    def create_ui(self):
        """Create simplified UI with radio button selection"""
        with gr.Blocks(theme=gr.themes.Default(), fill_height=True) as self.kiroku_agent:
            
            # ========================================
            # TAB 1: File Upload & Context
            # ========================================
            with gr.Tab("📁 File Context"):
                gr.Markdown("# Upload Configuration")
                
                with gr.Row():
                    file = gr.File(file_types=[".yaml"], scale=1)
                    js = gr.JSON(scale=5)
                
                gr.Markdown("### Discovered Files")
                _, _, files_preview = self._make_files_preview()
            
            # ========================================
            # TAB 2: Plot Generation with Radio Buttons
            # ========================================
            with gr.Tab("🎨 Plot Generation"):
                gr.Markdown("# Plot Generator")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        plot_prompt = gr.Textbox(
                            label="Plot Description",
                            placeholder="e.g., Create a sine wave with random noise overlay",
                            lines=3
                        )
                    
                    with gr.Column(scale=1):
                        num_plots_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=1,
                            step=1,
                            label="Number of variations"
                        )
                        
                        generate_btn = gr.Button(
                            "✨ Generate",
                            variant="primary",
                            size="lg"
                        )
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
                
                gr.Markdown("### Generated Plots")
                
                # Plot gallery (just images, no checkboxes)
                plot_images = []
                
                with gr.Row():
                    for i in range(NUM_PLOTS):
                        with gr.Column():
                            plot_img = gr.Plot(
                                label=f"Variation {i+1}",
                                visible=False
                            )
                            plot_images.append(plot_img)
                
                # Radio button for selection
                gr.Markdown("### Select Plot to View Code")
                plot_selector = gr.Radio(
                    choices=[],
                    label="Choose a plot",
                    value=None,
                    visible=False,
                    interactive=True
                )

                gr.Markdown("### Code for Selected Plot")
                selected_code = gr.Code(
                    label="Python Code",
                    language="python",
                    lines=15,
                    value="# Generate plots above to view code"
                )
            
            # ========================================
            # EVENT HANDLERS
            # ========================================
            
            # Upload YAML file
            file.upload(
                self.process_file,
                inputs=[file],
                outputs=[*files_preview, js]
            )
            
            # Generate plots button
            generate_btn.click(
                fn=self.generate_multiple_plots,
                inputs=[plot_prompt, num_plots_slider],
                outputs=[*plot_images, plot_selector, selected_code, status_text]
            )
            
            # Radio button selection
            plot_selector.change(
                fn=self.show_single_plot_code,
                inputs=[plot_selector],
                outputs=[selected_code, status_text]
            )
    
    def launch_ui(self):
        """Launch the Gradio interface"""
        logging.info(f"Working directory: {self.working_dir}")
        try:
            os.chdir(self.working_dir)
        except Exception as err:
            logging.warning(f"Directory {self.working_dir} does not exist: {err}")

        self.images = self.working_dir / "images"
        self.images.mkdir(parents=True, exist_ok=True)

        self.create_ui()
        self.kiroku_agent.launch()



def run():
    """Main entry point"""
    working_dir = Path(os.environ.get("KIROKU_PROJECT_DIRECTORY", "."))
    ui = SimplifiedKirokuUI(working_dir)
    ui.launch_ui()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    noisy_loggers = [
        "httpcore",
        "httpx",
        "urllib3",
        "matplotlib",
        "openai",
        "PIL",        # Image processing
        "multipart"   # File uploads
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    run()