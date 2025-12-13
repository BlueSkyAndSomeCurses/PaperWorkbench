import os
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from src.agents.states import AgentState, State
from src.agents.suggest_plot import PlotSuggester


@dataclass
class GeneratedPlotImage:
    """Metadata for a generated plot image"""
    path: str
    relative_path: str
    description: str
    rationale: str


class PlotGenerationAgent(State):
    """
    Execute approved plots locally (no CodeAPI needed).
    Uses PlotSuggester's execution capabilities.
    """
    
    def __init__(self, model: ChatOpenAI):
        super().__init__(model, "plot_generation")
        self.plotter = PlotSuggester(model)

    def run(self, state: AgentState) -> dict:
        """
        Execute approved plots and save images locally.
        """
        logging.info(f"state {self.name}: running")
        
        suggested_plots = state.suggested_plots
        if not suggested_plots:
            return {
                "state": self.name,
                "generated_plot_images": [],
                "draft": "No plots to generate.",
            }
        
        # Filter approved plots
        approved_plots = [p for p in suggested_plots if p.approved]
        
        if not approved_plots:
            return {
                "state": self.name,
                "generated_plot_images": [],
                "draft": f"No plots approved. {len(suggested_plots)} plots pending approval.",
            }
        
        logging.info(f"Generating {len(approved_plots)} approved plots...")
        
        generated_images: List[GeneratedPlotImage] = []
        
        # Set up output directory
        working_dir = Path(os.environ.get("KIROKU_PROJECT_DIRECTORY", "."))
        images_dir = working_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute each approved plot
        for i, plot in enumerate(approved_plots):
            try:
                logging.info(f"Executing plot {i + 1}/{len(approved_plots)}: {plot.description}")
                
                if plot.figure is not None:
                    fig = plot.figure
                    logging.info("Using pre-generated figure")
                else:
                    # Need to re-execute the code
                    logging.info("Re-executing plot code...")
                    fig = self.plotter._execute_plot_code(plot.code)
                
                # Save figure to file
                filename = self._generate_filename(plot, i)
                img_path = images_dir / filename
                
                fig.savefig(
                    img_path,
                    format='png',
                    bbox_inches='tight',
                    dpi=300,
                    facecolor='white',
                    edgecolor='none'
                )
                
                relative_path = f"images/{filename}"
                
                generated_images.append(GeneratedPlotImage(
                    path=str(img_path),
                    relative_path=relative_path,
                    description=plot.description,
                    rationale=plot.rationale
                ))
                
                logging.info(f"✓ Saved plot to: {img_path}")
                
            except Exception as e:
                logging.error(f"Failed to generate plot {i + 1}: {e}", exc_info=True)
                continue
        
        # Cleanup
        self.plotter.cleanup()
        
        # Create summary
        # summary = self._format_generation_summary(generated_images, len(approved_plots))
        
        return {
            "state": self.name,
            "generated_plot_images": generated_images
            # "draft": summary,
        }
    
    def _generate_filename(self, plot: 'PlotSuggestion', index: int) -> str:
        """Generate a descriptive filename for the plot."""
        import re
        
        if plot.filename_base:
            # Use source file name as base
            base = plot.filename_base.stem
        else:
            # Use sanitized description
            desc = plot.description[:30].lower()
            base = re.sub(r'[^a-z0-9]+', '_', desc).strip('_')
        
        return f"{base}_plot_{index + 1}.png"
    
    # def _format_generation_summary(self, images: List[GeneratedPlotImage], total: int) -> str:
    #     """Format generation results for display."""
    #     success_count = len(images)
        
    #     summary = f"## Plot Generation Complete\n\n"
    #     summary += f"**Results:** {success_count}/{total} plots generated successfully\n\n"
        
    #     if images:
    #         summary += "### Generated Images:\n\n"
    #         for img in images:
    #             summary += f"- **{img.description}**\n"
    #             summary += f"  - Path: `{img.relative_path}`\n"
    #             summary += f"  - ![Plot]({img.relative_path})\n\n"
        
    #     if success_count < total:
    #         summary += f"\n⚠️ Warning: {total - success_count} plot(s) failed to generate. Check logs.\n"
        
    #     return summary