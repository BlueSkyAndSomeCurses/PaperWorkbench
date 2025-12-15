import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from langchain_openai import ChatOpenAI
from decouple import config

from src.agents.states import RelevantFile
from src.agents.suggest_plot_base import PlotSuggester
from src.agents.prompts import VARIED_PLOT_PROMPT
from src.utils.constants import SUPPORTED_TABLE_FORMATS
from src.utils.models import PaperConfig

NUM_PLOTS = 2


@dataclass
class PlotVersion:
    """Represents a generated plot version"""
    id: str
    image_path: str | Path
    code: str
    timestamp: datetime
    selected: bool = False


class MinimalPlotUI:
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.plotter = None
        self.plot_gallery = []
        self.selected_plot_id = None
        self.state_values = None
        self.references = []

    def _initialize_plotter(self):
        """Initialize PlotSuggester"""
        try:
            model = ChatOpenAI(
                model="gpt-5-mini",
                temperature=1.0,
                api_key=config("OPENAI_API_KEY"),
            )
            plotter = PlotSuggester(model, self.working_dir)
            logging.info("✓ PlotSuggester initialized")
            return plotter
        except Exception as e:
            logging.error(f"Failed to initialize PlotSuggester: {e}")
            return None

    def read_initial_state(self, filename: Path) -> PaperConfig:
        """Reads initial state from a YAML file"""
        with filename.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logging.info(f"Initial yaml config: {data}")

        relevant_files = self._get_relevant_files(data)
        cfg = PaperConfig(**data, relevant_files=relevant_files)
        cfg.hypothesis = "\n\n".join(filter(None, [cfg.hypothesis, cfg.instructions]))

        return cfg

    def _get_relevant_files(self, state_values: dict) -> list[RelevantFile]:
        """Extract relevant files from working directory"""
        working_dir = Path(state_values.get("working_dir", self.working_dir))

        if not working_dir.exists() or not working_dir.is_dir():
            return []

        file_names_and_descs = {
            rel_file["file_name"]: rel_file.get("description", "")
            for rel_file in state_values.get("files_descriptions", [])
        }

        relevant_files = []
        for file in working_dir.rglob("*"):
            for file_name, description in file_names_and_descs.items():
                if (
                    file.suffix in [".yaml", ".yml", ".md", ".tex", ".html", *SUPPORTED_TABLE_FORMATS]
                    and file.name == file_name
                ):
                    relevant_files.append(
                        RelevantFile(file_path=file, description=description)
                    )

        return relevant_files

    def process_file(self, filename: str):
        """Process uploaded YAML file"""
        try:
            pwd = Path.cwd()
            logging.info(f"Processing file: {filename}")
            
            file_path = pwd / Path(filename).name
            self.state_values = self.read_initial_state(file_path)
            self.working_dir = self.state_values.working_dir
            
            files_preview = self._update_files_preview()
            
            # Ensure we return exactly 22 values (20 previews + config + status)
            return (
                *files_preview,  # 20 values
                self.state_values.model_dump_json(),  # config_json
                gr.update(value="✓ File loaded successfully", visible=True)  # file_status
            )
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            # Return 22 values on error too
            empty_previews = [gr.update(visible=False)] * 20
            return (
                *empty_previews,
                f'{{"error": "{str(e)}"}}',
                gr.update(value=f"❌ Error: {str(e)}", visible=True)
            )

    # def generate_multiple_plots(self, plot_prompt: str, num_plots: int):
    #     """Generate multiple plot variations"""
    #     try:
    #         if not self.plotter:
    #             self.plotter = self._initialize_plotter()

    #         if not self.plotter:
    #             yield self._create_error_message("Could not initialize plotter.")
    #             return

    #         self.plot_gallery = []
    #         self.selected_plot_id = None

    #         # Get paper context
    #         paper_context = ""
    #         relevant_files = []

    #         if self.state_values:
    #             relevant_files = self.state_values.relevant_files
    #             if hasattr(self.state_values, "hypothesis"):
    #                 paper_context = self.state_values.hypothesis

    #         has_prompt = bool(plot_prompt.strip())
    #         has_draft = bool(paper_context.strip())

    #         if not has_prompt and not has_draft and not relevant_files:
    #             yield self._create_error_message("⚠️ Please provide a plot description.")
    #             return

    #         # Generate plots
    #         for relevant_file in relevant_files:
    #             logging.info(f"Generating plot for: {relevant_file.file_path}")

    #             for i in range(num_plots):
    #                 logging.info(f"Generating variation {i + 1}/{num_plots}...")

    #                 try:
    #                     varied_prompt = plot_prompt
    #                     if num_plots > 1:
    #                         varied_prompt = VARIED_PLOT_PROMPT.format(
    #                             plot_prompt=plot_prompt,
    #                             iteration=i,
    #                             num_plots=num_plots,
    #                         )

    #                     fig_path, code = self.plotter.suggest_plot(
    #                         relevant_file=relevant_file,
    #                         paper_content=paper_context if has_draft else "",
    #                         user_prompt=varied_prompt,
    #                         num_plots=num_plots,
    #                     )

    #                     plot_version = PlotVersion(
    #                         id=str(uuid.uuid4()),
    #                         image_path=fig_path,
    #                         code=code,
    #                         timestamp=datetime.now(),
    #                         selected=False,
    #                     )

    #                     self.plot_gallery.append(plot_version)
    #                     logging.info(f"✓ Variation {i + 1} generated")

    #                 except Exception as e:
    #                     logging.error(f"Failed to generate variation {i + 1}: {e}")
    #                     continue

    #             success_count = len(self.plot_gallery)
    #             logging.info(f"✓ Complete: {success_count}/{num_plots} successful")

    #             yield self.render_gallery()

    def generate_multiple_plots(self, plot_prompt: str):  # ← Remove num_plots parameter
        """Generate plots based on prompt and available files"""
        try:
            if not self.plotter:
                self.plotter = self._initialize_plotter()
            if not self.plotter:
                yield self._create_error_message("Could not initialize plotter.")
                return

            self.plot_gallery = []
            self.selected_plot_id = None

            # Get paper context
            paper_context = ""
            relevant_files = []

            if self.state_values:
                relevant_files = self.state_values.relevant_files
                if hasattr(self.state_values, "hypothesis"):
                    paper_context = self.state_values.hypothesis

            has_prompt = bool(plot_prompt.strip())
            has_draft = bool(paper_context.strip())

            if not relevant_files:
                yield self._create_error_message("⚠️ No data files found. Please upload a config with data files.")
                return

            # Generate ONE plot per file
            for relevant_file in relevant_files:
                logging.info(f"Generating plot for: {relevant_file.file_path}")
                
                # REMOVE: for i in range(num_plots):  ← Delete this entire inner loop

                try:
                    # Build context-aware prompt
                    if has_prompt:
                        # User specified what they want
                        final_prompt = f"{plot_prompt}\n\nContext: Using data from {relevant_file.file_path.name}"
                        if relevant_file.description:
                            final_prompt += f"\nFile purpose: {relevant_file.description}"
                    else:
                        # Auto-generate based on file description and paper context
                        final_prompt = f"Create a publication-quality plot"
                        if relevant_file.description:
                            final_prompt += f" showing: {relevant_file.description}"

                    # REMOVE: VARIED_PLOT_PROMPT logic entirely
                    # REMOVE: varied_prompt = ...

                    fig_path, code = self.plotter.suggest_plot(
                        relevant_file=relevant_file,
                        paper_content=paper_context if has_draft else "",
                        user_prompt=final_prompt,  # ← Changed variable name
                        num_plots=1,  # ← Always 1 now
                    )

                    plot_version = PlotVersion(
                        id=str(uuid.uuid4()),
                        image_path=fig_path,
                        code=code,
                        timestamp=datetime.now(),
                        selected=False,
                    )

                    self.plot_gallery.append(plot_version)
                    logging.info(f"✓ Plot generated for {relevant_file.file_path.name}")

                except Exception as e:
                    logging.error(f"Failed to generate plot for {relevant_file.file_path.name}: {e}")
                    continue

            success_count = len(self.plot_gallery)
            total_files = len(relevant_files)
            logging.info(f"✓ Complete: {success_count}/{total_files} plots generated")

            yield self.render_gallery()

        except Exception as e:
            logging.error("Critical error in generate_multiple_plots", exc_info=True)
            yield self._create_error_message(f"Critical Error:\n{str(e)}")
        finally:
            if self.plotter:
                self.plotter.cleanup()


    def render_gallery(self):
        """Render the plot gallery"""
        if not self.plot_gallery:
            empty_updates = [gr.update(visible=False)] * NUM_PLOTS
            return (
                *empty_updates,
                gr.update(choices=[], value=None),
                "# No plots generated",
                "Ready",
            )

        plot_updates = []
        for i in range(NUM_PLOTS):
            if i < len(self.plot_gallery):
                plot = self.plot_gallery[i]
                plot_updates.append(gr.update(value=str(plot.image_path), visible=True))
            else:
                plot_updates.append(gr.update(visible=False))

        # Better radio labels - show file names if available
        radio_choices = []
        for i, plot in enumerate(self.plot_gallery):
            # ← FIX: Check if image_path is actually a path
            try:
                if isinstance(plot.image_path, (str, Path)):
                    filename = Path(plot.image_path).stem.replace('_plot_', ' - ')
                    radio_choices.append(f"Plot {i + 1}: {filename}")
                else:
                    # Fallback if it's not a path
                    radio_choices.append(f"Plot {i + 1}")
            except Exception:
                radio_choices.append(f"Plot {i + 1}")

        radio_update = gr.update(
            choices=radio_choices,
            value=radio_choices[0] if radio_choices else None,
            visible=True,
        )

        code = self.plot_gallery[0].code if self.plot_gallery else "# No code"

        num_plots = len(self.plot_gallery)
        status = f"✓ {num_plots} plot{'s' if num_plots != 1 else ''} generated (1 per data file)"

        return (*plot_updates, radio_update, code, status)

    def show_single_plot_code(self, plot_choice: str):
        """Show code for selected plot"""
        if not plot_choice or not self.plot_gallery:
            return "# No plot selected", "No selection"

        try:
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
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
        ax.axis("off")

        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        error_updates = [gr.update(value=image_array, visible=True)]
        error_updates.extend([gr.update(visible=False)] * (NUM_PLOTS - 1))

        return (
            *error_updates,
            gr.update(choices=[], value=None),
            f"# Error\n{message}",
            message,
        )



    def _update_files_preview(self) -> list:
        """Update file preview components"""
        relevant_files = self.state_values.relevant_files
        updates = []

        for i in range(20):  # Support up to 20 files
            if i < len(relevant_files):
                file = relevant_files[i].file_path
                try:
                    if file.suffix in [".csv"]:
                        # Read CSV and convert to markdown table
                        table = pl.scan_csv(str(file)).head(5).collect()
                        markdown_table = f"**{file.name}** (first 5 rows)\n\n{table.to_pandas().to_markdown()}"
                        updates.append(gr.update(value=markdown_table, visible=True))
                    elif file.suffix in [".parquet"]:
                        # Read Parquet and convert to markdown table
                        table = pl.scan_parquet(str(file)).head(5).collect()
                        markdown_table = f"**{file.name}** (first 5 rows)\n\n{table.to_pandas().to_markdown()}"
                        updates.append(gr.update(value=markdown_table, visible=True))
                    elif file.suffix in [".html"]:
                        # For HTML files, show raw content (truncated)
                        content = file.read_text()[:1000]
                        preview = f"**{file.name}**\n\n```html\n{content}\n...\n```"
                        updates.append(gr.update(value=preview, visible=True))
                    elif file.suffix in [".yaml", ".yml"]:
                        # For YAML files
                        content = file.read_text()[:1000]
                        preview = f"**{file.name}**\n\n```yaml\n{content}\n...\n```"
                        updates.append(gr.update(value=preview, visible=True))
                    elif file.suffix in [".md"]:
                        # For Markdown files
                        content = file.read_text()[:1000]
                        preview = f"**{file.name}**\n\n{content}\n\n..."
                        updates.append(gr.update(value=preview, visible=True))
                    elif file.suffix in [".tex"]:
                        # For LaTeX files
                        content = file.read_text()[:1000]
                        preview = f"**{file.name}**\n\n```latex\n{content}\n...\n```"
                        updates.append(gr.update(value=preview, visible=True))
                    elif file.suffix in [".jpeg", ".jpg", ".png"]:
                        # For images, just show filename
                        preview = f"**{file.name}**\n\n📷 Image file: `{file.name}`"
                        updates.append(gr.update(value=preview, visible=True))
                    else:
                        # For other files, just show the filename
                        preview = f"**{file.name}**\n\nFile type: `{file.suffix}`"
                        updates.append(gr.update(value=preview, visible=True))
                except Exception as e:
                    logging.error(f"Error loading {file.name}: {e}")
                    error_msg = f"**{file.name}**\n\n❌ Error: {str(e)}"
                    updates.append(gr.update(value=error_msg, visible=True))
            else:
                updates.append(gr.update(visible=False))

        return updates

    def create_ui(self):
        """Create minimal Gradio UI"""
        with gr.Blocks(theme=gr.themes.Default()) as app:
            gr.Markdown("# 🎨 Minimal Plot Generator")
            
            with gr.Tab("📁 File Upload"):
                file_upload = gr.File(file_types=[".yaml"], label="Upload YAML Config")
                file_status = gr.Textbox(label="Status", visible=False)
                config_json = gr.JSON(label="Loaded Configuration")
                
                gr.Markdown("### File Previews")
                # Use Markdown for all file previews (can display both text and tables)
                file_previews = [
                    gr.Markdown(visible=False, label=f"File {i+1}")
                    for i in range(20)
                ]

            with gr.Tab("🎨 Plot Generation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        plot_prompt = gr.Textbox(
                            label="Plot Description",
                            placeholder="Describe the plot you want to create",
                            lines=3,
                        )
                    with gr.Column(scale=1):
                        # num_plots_slider = gr.Slider(
                        #     minimum=1, maximum=5, value=1, step=1,
                        #     label="Number of variations",
                        # )
                        generate_btn = gr.Button("✨ Generate", variant="primary", size="lg")

                status_text = gr.Textbox(label="Status", value="Ready", interactive=False)

                gr.Markdown("### Generated Plots")
                with gr.Row():
                    plot_images = []
                    for i in range(NUM_PLOTS):
                        with gr.Column():
                            plot_img = gr.Image(
                                label=f"Variation {i + 1}",
                                visible=False,
                                type="filepath"
                            )
                            plot_images.append(plot_img)

                gr.Markdown("### Select Plot to View Code")
                plot_selector = gr.Radio(
                    choices=[], label="Choose a plot",
                    value=None, visible=False, interactive=True,
                )

                gr.Markdown("### Code for Selected Plot")
                selected_code = gr.Code(
                    label="Python Code", language="python", lines=15,
                    value="# Generate plots above to view code",
                )

            # Event handlers
            file_upload.upload(
                self.process_file,
                file_upload,
                [*file_previews, config_json, file_status]
            )

            generate_btn.click(
                self.generate_multiple_plots,
                # [plot_prompt, num_plots_slider],
                [plot_prompt],
                [*plot_images, plot_selector, selected_code, status_text],
            )

            plot_selector.change(
                self.show_single_plot_code,
                plot_selector,
                [selected_code, status_text],
            )

        return app

    def launch(self):
        """Launch the UI"""
        self.images = self.working_dir / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        
        app = self.create_ui()
        app.queue().launch()


def run():
    """Main entry point"""
    working_dir = Path(os.environ.get("KIROKU_PROJECT_DIRECTORY", "."))
    ui = MinimalPlotUI(working_dir)
    ui.launch()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    run()