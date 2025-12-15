"""
Minimal Kiroku UI - Plot Generation Testing Only
Fast, lightweight version for testing plot generation without document workflow
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid

import gradio as gr
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from decouple import config

from src.agents.suggest_plot import PlotSuggester

logging.basicConfig(level=logging.DEBUG)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

NUM_PLOTS = 5


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


class MinimalKirokuUI:
    """Minimal UI for plot generation testing"""
    
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.plotter = self._initialize_plotter()
        self.plot_gallery = []
        self.selected_plot_ids = set()
        logging.info("✓ Minimal UI initialized")

    def _initialize_plotter(self):
        """Initialize PlotSuggester"""
        try:
            model = ChatOpenAI(
                model=config("OPENAI_MODEL_NAME", default="gpt-5-nano"),
                temperature=1.0,
                openai_api_key=config("OPENAI_API_KEY"),
            )
            plotter = PlotSuggester(model)
            logging.info("✓ PlotSuggester initialized")
            return plotter
        except Exception as e:
            logging.error(f"Failed to initialize PlotSuggester: {e}")
            return None

    def generate_multiple_plots(self, plot_prompt: str, num_plots: int):
        """Generate multiple plot variations"""
        try:
            if not plot_prompt.strip():
                return self._create_error_message("⚠️ Please enter a plot description")

            logging.info(f"Generating {num_plots} plot(s)")

            # Clear previous gallery
            self.plot_gallery = []
            self.selected_plot_ids = set()

            # Generate plots
            for i in range(num_plots):
                logging.info(f"Generating plot {i+1}/{num_plots}...")
                
                try:
                    fig, code = self.plotter.suggest_plot(
                        paper_content="",
                        user_prompt=plot_prompt,
                        num_plots=num_plots
                    )

                    plot_version = PlotVersion(
                        id=str(uuid.uuid4()),
                        figure=fig,
                        code=code,
                        timestamp=datetime.now(),
                        selected=False
                    )

                    self.plot_gallery.append(plot_version)
                    logging.info(f"✓ Plot {i+1} generated")

                except Exception as e:
                    logging.error(f"Failed to generate plot {i+1}: {e}")
                    continue

            success_count = len(self.plot_gallery)
            logging.info(f"✓ Complete: {success_count}/{num_plots} successful")

            return self.render_gallery(num_plots)

        except Exception as e:
            logging.error(f"Critical error: {e}", exc_info=True)
            return self._create_error_message(f"Error: {str(e)}")
        finally:
            if self.plotter:
                self.plotter.cleanup()

    def render_gallery(self, num_plots):
        """Render the gallery display"""
        if not self.plot_gallery:
            empty_updates = [gr.update(visible=False)] * (num_plots * 2)
            return (
                *empty_updates,
                "# No plots generated",
                "Ready"
            )

        gallery_updates = []
        for i in range(NUM_PLOTS):
            if i < len(self.plot_gallery):
                plot = self.plot_gallery[i]
                gallery_updates.extend([
                    gr.update(value=plot.figure, visible=True),
                    gr.update(value=plot.selected, visible=True, interactive=True),
                ])
            else:
                gallery_updates.extend([
                    gr.update(visible=False),
                    gr.update(visible=False),
                ])

        code = self.get_selected_plots_code()
        num_selected = len(self.selected_plot_ids)
        status = f"✓ {len(self.plot_gallery)} plots | Selected: {num_selected}"

        return (*gallery_updates, code, status)

    def handle_checkbox_change(self, slot_idx: int, is_checked: bool):
        """Handle checkbox selection"""
        if slot_idx < len(self.plot_gallery):
            plot = self.plot_gallery[slot_idx]
            if is_checked:
                self.selected_plot_ids.add(plot.id)
                plot.selected = True
            else:
                self.selected_plot_ids.discard(plot.id)
                plot.selected = False

            logging.info(f"Plot {slot_idx+1} {'selected' if is_checked else 'deselected'}")

        return self.update_selection_display()

    def update_selection_display(self):
        """Update selection display"""
        code = self.get_selected_plots_code()
        num_selected = len(self.selected_plot_ids)
        status = f"✓ {len(self.plot_gallery)} plots | Selected: {num_selected}"

        checkbox_updates = []
        for i in range(NUM_PLOTS):
            if i < len(self.plot_gallery):
                plot = self.plot_gallery[i]
                checkbox_updates.append(gr.update(value=plot.selected))
            else:
                checkbox_updates.append(gr.update())

        return (*checkbox_updates, code, status)

    def get_selected_plots_code(self):
        """Get code for selected plots"""
        if not self.selected_plot_ids:
            return "# No plots selected\n# Check boxes above to select plots"

        selected_plots = [p for p in self.plot_gallery if p.id in self.selected_plot_ids]

        if not selected_plots:
            return "# No plots selected"

        code_blocks = ["# Selected Plots\n"]
        for idx, plot in enumerate(selected_plots, 1):
            code_blocks.append(f"\n# ===== Plot {idx} =====\n")
            code_blocks.append(plot.code)

        return "\n".join(code_blocks)

    def _create_error_message(self, message: str):
        """Create error display"""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=11)
        ax.axis('off')

        error_updates = []
        for i in range(NUM_PLOTS):
            if i == 0:
                error_updates.extend([
                    gr.update(value=fig, visible=True),
                    gr.update(visible=False)
                ])
            else:
                error_updates.extend([
                    gr.update(visible=False),
                    gr.update(visible=False)
                ])

        return (
            *error_updates,
            f"# Error\n{message}",
            f"❌ {message}"
        )

    def create_ui(self):
        """Create minimal UI"""
        with gr.Blocks(theme=gr.themes.Default()) as self.kiroku_agent:
            
            gr.Markdown("# 🎨 Plot Generator Test")
            
            with gr.Row():
                with gr.Column(scale=2):
                    plot_prompt = gr.Textbox(
                        label="Plot Description",
                        placeholder="e.g., Create a sine wave with random noise overlay",
                        lines=3
                    )
                    gr.Markdown("""
                    **Examples:**
                    - "Show correlation between x and y with regression line"
                    - "Create bar chart comparing values across categories"
                    - "Plot time series with trend line"
                    """)
                
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
                value="Ready! Enter a description above",
                interactive=False
            )
            
            gr.Markdown("### Generated Plots")
            
            # Plot gallery
            plot_components = []
            plot_checkboxes = []
            
            with gr.Row():
                for i in range(NUM_PLOTS):
                    with gr.Column():
                        with gr.Group():
                            plot_img = gr.Plot(label=f"Variation {i+1}", visible=False)
                            plot_checkbox = gr.Checkbox(
                                label="✓ Select",
                                value=False,
                                visible=False
                            )
                            plot_components.extend([plot_img, plot_checkbox])
                            plot_checkboxes.append(plot_checkbox)
            
            gr.Markdown("### Selected Code")
            selected_code = gr.Code(
                label="Python Code",
                language="python",
                lines=15,
                value="# Select plots above to view code"
            )
            
            # Wire events
            generate_btn.click(
                fn=self.generate_multiple_plots,
                inputs=[plot_prompt, num_plots_slider],
                outputs=[*plot_components, selected_code, status_text]
            )
            
            for idx, checkbox in enumerate(plot_checkboxes):
                checkbox.change(
                    fn=lambda checked, idx=idx: self.handle_checkbox_change(idx, checked),
                    inputs=[checkbox],
                    outputs=[*plot_checkboxes, selected_code, status_text]
                )

    def launch_ui(self):
        """Launch the UI"""
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.create_ui()
        self.kiroku_agent.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False
        )


def run():
    """Main entry point"""
    working_dir = Path(os.environ.get("KIROKU_PROJECT_DIRECTORY", "."))
    ui = MinimalKirokuUI(working_dir)
    ui.launch_ui()


if __name__ == "__main__":
    run()