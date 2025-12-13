# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import logging
import os
import re
import shutil
from langgraph.managed.base import Type
from numpy import int16
import pandas as pd
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Set, Optional
import uuid

import gradio as gr
import matplotlib
import markdown
import yaml
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.states import *
from src.agents.suggest_plot import PlotSuggester
import matplotlib.pyplot as plt

from src.utils.models import PaperConfig
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from decouple import config

logging.basicConfig(level=logging.WARNING)

NUM_PLOTS= 5

class DocumentWriter:
    def __init__(
        self,
        suggest_title: bool = False,
        generate_citations: bool = True,
        model_name: str = "openai",
        temperature: float = 0.0,
    ) -> None:
        self.suggest_title = suggest_title
        self.generate_citations = generate_citations
        self.state = None
        self.set_thread_id(1)

        # TODO make models configurable for different tasks
        self.model_m = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=config("OPENAI_API_KEY"),
        )
        self.state_nodes = {
            node.name: node
            for node in [
                SuggestTitle(self.model_m),
                SuggestTitleReview(self.model_m),
                InternetSearch(self.model_m),
                TopicSentenceWriter(self.model_m),
                TopicSentenceManualReview(self.model_m),
                PaperWriter(self.model_m),
                WriterManualReviewer(self.model_m),
                ReflectionReviewer(self.model_m),
                ReflectionManualReview(self.model_m),
                WriteAbstract(self.model_m),
                GenerateReferences(self.model_m),
                GenerateCitations(self.model_m),
                GenerateFigureCaptions(self.model_m),
                LaTeXConverter(self.model_m),
            ]
            if self.mask_nodes(node.name)
        }
        self.create_graph()

    def mask_nodes(self, name: str) -> bool:
        """
        We do not process nodes if user does not want to run that phase.
        :param name: name of the node.
        :return: True if we keep nodes, False otherwise
        """
        if name in ["suggest_title", "suggest_title_review"] and not self.suggest_title:
            return False
        return not (
            name in ["generate_references", "generate_citations"]
            and not self.generate_citations
        )

    def create_graph(self) -> None:
        """
        Builds a graph to execute the different phases of a document writing.

        :return: Nothing.
        """
        memory = MemorySaver()

        builder = StateGraph(AgentState)

        # Add nodes to the graph
        for name, state in self.state_nodes.items():
            builder.add_node(name, state.run)

        # Add edges to the graph
        if self.suggest_title:
            builder.add_conditional_edges(
                "suggest_title_review",
                self.is_title_review_complete,
                {"next_phase": "internet_search", "review_more": "suggest_title"},
            )
        builder.add_conditional_edges(
            "topic_sentence_manual_review",
            self.is_plan_review_complete,
            {
                "topic_sentence_manual_review": "topic_sentence_manual_review",
                "paper_writer": "paper_writer",
            },
        )

        builder.add_conditional_edges(
            "writer_manual_reviewer",
            self.is_generate_review_complete,
            {
                "manual_review": "writer_manual_reviewer",
                "reflection": "reflection_reviewer",
                "finalize": "write_abstract",
            },
        )
        if self.suggest_title:
            builder.add_edge("suggest_title", "suggest_title_review")
        builder.add_edge("internet_search", "topic_sentence_writer")
        builder.add_edge("topic_sentence_writer", "topic_sentence_manual_review")
        builder.add_edge("paper_writer", "writer_manual_reviewer")
        builder.add_edge("reflection_reviewer", "additional_reflection_instructions")
        builder.add_edge("additional_reflection_instructions", "paper_writer")

        if self.generate_citations:
            builder.add_edge("write_abstract", "generate_references")
            builder.add_edge("generate_references", "generate_citations")
            builder.add_edge("generate_citations", "generate_figure_captions")
        else:
            builder.add_edge("write_abstract", "generate_figure_captions")
        builder.add_edge("generate_figure_captions", "latex_converter")
        builder.add_edge("latex_converter", END)

        # Starting state is either suggest_title or planner.
        if self.suggest_title:
            builder.set_entry_point("suggest_title")
        else:
            builder.set_entry_point("internet_search")

        self.interrupt_after = []
        self.interrupt_before = ["suggest_title_review"] if self.suggest_title else []
        self.interrupt_before.extend(
            [
                "topic_sentence_manual_review",
                "writer_manual_reviewer",
                "additional_reflection_instructions",
            ]
        )
        if self.generate_citations:
            self.interrupt_before.append("generate_citations")
        # Build graph
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_before=self.interrupt_before,
            interrupt_after=self.interrupt_after,
            debug=False,
        )

    def is_title_review_complete(self, state: AgentState) -> str:
        """
        Checks if title review is complete based on an END instruction.

        :param state: state of agent.
        :return: next state of agent.
        """

        if not state["messages"]:
            return "next_phase"
        return "review_more"

    def is_plan_review_complete(self, state: AgentState, config: dict) -> str:
        """
        Checks if plan manual review is complete based on an empty instruction.

        :param state: state of agent.
        :return: next state of agent.
        """
        if config["configurable"]["instruction"]:
            return "topic_sentence_manual_review"
        return "paper_writer"

    def is_generate_review_complete(self, state: AgentState, config: dict) -> str:
        """
        Checks if review of generation phase is complete based on number of revisions.

        :param state: state of agent.
        :return: next state to go.
        """
        if config["configurable"]["instruction"]:
            return "manual_review"
        if state["revision_number"] <= state["max_revisions"]:
            return "reflection"
        return "finalize"

    def invoke(self, state: PaperConfig, config: dict) -> str:
        """
        Invokes the multi-agent system to write a paper.

        :param state: state of initial invokation.
        :return: draft
        """
        config = {"configurable": config}
        config["configurable"]["thread_id"] = self.get_thread_id()
        response = self.graph.invoke(state, config)
        self.state = response
        draft = response.get("draft", "").strip()
        # we have to do this because the LLM sometimes decide to add
        # this to the final answer.
        if "```markdown" in draft:
            draft = "\n".join(draft.split("\n")[1:-1])
        return draft

    def stream(self, state, config):
        """
        Invokes the multi-agent system to write a paper.

        :param state: state of initial invokation.
        :return: full state information
        """
        config = {"configurable": config}
        config["configurable"]["thread_id"] = self.get_thread_id()
        for event in self.graph.stream(state, config, stream_mode="values"):
            pass
        draft = event["draft"]
        # we have to do this because the LLM sometimes decide to add
        # this to the final answer.
        if "```markdown" in draft:
            draft = "\n".join(draft.split("\n")[1:-1])
        return draft

    def get_state(self):
        """
        Returns the full state of the document writing process.
        :return: Generated state from invoke
        """
        config = {"configurable": {"thread_id": self.get_thread_id()}}
        return self.graph.get_state(config)

    def update_state(self, new_state):
        """
        Updates the state of langgraph.
        :param new_state:
        :return: None
        """
        config = {"configurable": {"thread_id": self.get_thread_id()}}
        self.graph.update_state(config, new_state.values)

    def get_thread_id(self):
        return str(self.thread_id)

    def set_thread_id(self, thread_id):
        self.thread_id = str(thread_id)

    def draw(self):
        display(
            Image(
                self.graph.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API
                )
            )
        )

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

class KirokuUI:
    def __init__(self,
                 working_dir: Path,
                 model_name: str = "gpt-5-nano",
                 temperature: float = 1):
        self.working_dir = working_dir
        self.first = True
        self.next_state = -1
        self.references = []
        self.state_values = PaperConfig()
        self.writer = None
        self.model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=config("OPENAI_API_KEY"),
            )
        self.plotter = self._initialize_plotter()

        self.plot_gallery = []
        self.selected_plot_ids = set()
    def _initialize_plotter(self):
        """
        Initialize PlotSuggester independently of DocumentWriter.
        This allows plot generation without uploading YAML or generating paper.
        """
        try:
            plotter = PlotSuggester(self.model)
            logging.info("✓ PlotSuggester initialized independently")
            return plotter
        except Exception as e:
            logging.error(f"Failed to initialize PlotSuggester: {e}")
            return None

    def read_initial_state(self, filename: Path) -> PaperConfig:
        """
        Reads initial state from a YAML 'filename'.
        :param filename: YAML file containing initial paper configuration.
        :return: initial state dictionary.
        """
        with filename.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logging.info(data)
        cfg = PaperConfig(**data)
        cfg.hypothesis = "\n\n".join(filter(None, [cfg.hypothesis, cfg.instructions]))

        return cfg

    def step(self, instruction: str, state_values: PaperConfig = None) -> str:
        """
        Performs one step of the graph invocation, stopping at the next break point.
        :param instruction: instruction to execute.
        :param state_values: initial state values or None if continuing.
        :return: draft of the paper.
        """
        config = {"instruction": instruction}
        return self.writer.invoke(state_values, config)

    def update(self, instruction):
        """
        Updates state upon submitting an instruction or updating references.
        :param instruction: instruction to be executed.
        :return: new draft, atlas message and making input object non-interactive.
        """
        draft = self.step(instruction)
        state = self.writer.get_state()
        current_state = state.values["state"]
        try:
            next_state = state.next[0]
        except:
            next_state = "NONE"

        # if state is in reflection stage, draft to be shown is in the critique field.
        if (
            current_state == "reflection_reviewer"
            and next_state == "additional_reflection_instructions"
        ):
            draft = state.values["critique"]

        # if next state is going to generate citations, we populate the references
        # for the Tab References.
        if next_state == "generate_citations":
            self.references = state.values.get("references", []).split("\n")

        # if we have reached the end, we will save everything.
        if next_state == END or next_state == "NONE":
            dir = os.path.splitext(self.filename)[0]
            logging.warning(f"saving final draft in {dir}")
            self.save_as()

        self.next_state = next_state
        return (draft, self.atlas_message(next_state), gr.update(interactive=False))

    def atlas_message(self, state):
        """
        Returns the Echo message for a given state.
        :param state: Next state of the multi-agent system.
        :return:
        """
        message = {
            "suggest_title_review": "Please suggest review instructions for the title.",
            "topic_sentence_manual_review": "Please suggest review instructions for the topic sentences.",
            "writer_manual_reviewer": "Please suggest review instructions for the main draft.",
            "additional_reflection_instructions": "Please provide additional instructions for the overall paper review.",
            "generate_citations": "Please look at the references tab and confirm the references.",
        }

        instruction = message.get(state, "")
        if instruction or state == "generate_citations":
            if state == "generate_citations":
                return instruction
            return instruction + " Type <RETURN> when done."
        return "We have reached the end."

    def initial_step(self) -> tuple[str, str]:
        """
        Performs initial step, in which we need to providate a state to the graph.
        :return: draft and Echo message.
        """
        state_values = deepcopy(self.state_values)
        if self.state_values.suggest_title:
            state_values.state = "suggest_title"
        else:
            state_values.state = "topic_sentence_writer"
        # initialize a bunch of variables users should not care about.
        # in principle this could be initialized in the Pydantic object,
        # but I could not make this work there.
        draft = self.step("", state_values)
        state = self.writer.get_state()
        current_state = state.values["state"]
        try:
            next_state = state.next[0]
        except:
            next_state = "NONE"
        return draft, self.atlas_message(next_state)

    def process_file(self, filename: str):
        """
        Processes file uploaded.
        :param filename: file name where to read the file.
        :return: State that was read and make input non-interactive.
        """
        pwd = Path.cwd()

        logging.warning(f"Setting working directory to {pwd}, file uploaded {filename}")

        self.filename = pwd / Path(filename).name

        self.state_values = self.read_initial_state(Path(filename))

        if self.state_values:
            suggest_title, generate_citations, model_name, temperature = (
                self.state_values.suggest_title,
                self.state_values.generate_citations,
                self.state_values.model_name,
                self.state_values.temperature,
            )
            self.writer = DocumentWriter(
                suggest_title=suggest_title,
                generate_citations=generate_citations,
                model_name=model_name,
                temperature=temperature,
            )
            self.plotter = PlotSuggester(self.writer.model_m)

        return self.state_values, gr.update(interactive=False)

    def save_as(self):
        """
        Saves project status. We save all instructions given by the user.
        :return: message where the project was saved.
        """
        filename = self.filename
        state = self.writer.get_state()

        print(state)

        draft = state.values.get("draft", "")
        latex_draft = state.values.get("latex_draft", "")
        # need to replace file= by empty because of gradio problem in Markdown
        draft = re.sub(r"\/?file=", "", draft)
        plan = state.values.get("plan", "")
        review_topic_sentences = "\n\n".join(
            state.values.get("review_topic_sentences", [])
        )
        review_instructions = "\n\n".join(state.values.get("review_instructions", []))
        content = "\n\n".join(state.values.get("content", []))

        dir_ = filename.parent / filename.stem

        dir_.mkdir(exist_ok=True, parents=True)

        try:
            (dir_ / "images").symlink_to(self.images)
        except Exception as err:
            logging.error(f"Error occurred while creating symlink for images {err}")
        base_filename = str(dir_ / dir_.name)
        logging.info(f"Saving to {base_filename}")
        with open(base_filename + ".md", "w") as fp:
            fp.write(draft)
            logging.warning(f"saved file {base_filename + '.md'}")

        html = markdown.markdown(draft)
        with open(base_filename + ".html", "w") as fp:
            fp.write(html)

        with open(base_filename + ".tex", "w") as fp:
            fp.write(latex_draft)

        try:
            # Use pandoc to convert to docx
            subprocess.run(
                [
                    "pandoc",
                    "-s",
                    f"{base_filename + '.html'}",
                    "-f",
                    "html",
                    "-t",
                    "docx",
                    "-o",
                    f"{base_filename + '.docx'}",
                ]
            )

            subprocess.run(
                [
                    "pandoc",
                    "-s",
                    f"{base_filename + '.md'}",
                    "-f",
                    "markdown",
                    "-t",
                    "latex",
                    "-o",
                    f"{base_filename + 'pandoc.tex'}",
                ]
            )
            subprocess.run(
                [
                    "pandoc",
                    "-s",
                    f"{base_filename + '.md'}",
                    "-o",
                    f"{base_filename + '.pdf'}",
                    "--pdf-engine=pdflatex",
                ]
            )
        except:
            logging.error("cannot find 'pandoc'")

        # with open(base_filename + ".docx", "wb") as fp:
        #    buf = html2docx(html, title=state.values.get("title", ""))
        #    fp.write(buf.getvalue())
        logging.warning(f"saved file {base_filename + '.docx'}")

        with open(base_filename + "_ts.txt", "w") as fp:
            fp.write(review_topic_sentences)
            logging.warning(f"saved file {base_filename + '_ts.txt'}")

        with open(base_filename + "_wi.txt", "w") as fp:
            fp.write(review_instructions)
            logging.warning(f"saved file {base_filename + '_wi.txt'}")

        with open(base_filename + "_plan.md", "w") as fp:
            fp.write(plan)
            logging.warning(f"saved file {base_filename + '_plan.md'}")

        with open(base_filename + "_content.txt", "w") as fp:
            fp.write(content)
            logging.warning(f"saved file {base_filename + '_content.txt'}")

        return f"Saved project {dir}"

    def update_refs(self) -> list:
        """
        Updates the reference for Gradio
        :return: list of gr.update objects.
        """
        ref_list = [
            gr.update(value=True, visible=True, label=reference)
            for i, reference in enumerate(self.references)
        ] + [gr.update()] * (1000 - len(self.references))
        return [
            gr.update(
                visible=self.state_values.generate_citations
                and len(self.references) > 0
            ),
            *ref_list,
        ]

    def submit_ref_list(self, *ref_list):
        """
        Invokes step of generating citations with user reference feedback.
        :param ref_list: List of references that were unselected.
        :return: Everything returned by self.update.
        """
        ref_list = ref_list[: len(self.references)]
        state = self.writer.get_state()
        references = [
            self.references[i] for i in range(len(self.references)) if ref_list[i]
        ]
        logging.warning("Keeping the following references")
        for ref in references:
            logging.warning(ref)
        state.values["references"] = "\n".join(references)
        self.writer.update_state(state)
        return self.update("")

    def toggle_plot_selection(self, plot_id: str):
        """Toggle selection state of a plot"""
        if plot_id in self.selected_plot_ids:
            self.selected_plot_ids.remove(plot_id)
        else:
            self.selected_plot_ids.add(plot_id)
        for plot in self.plot_gallery:
            if plot.id == plot_id:
                plot.selected = plot_id in self.selected_plot_ids
                break
        
        return self.update_selection_display()

    def get_selected_plots_code(self):
        """Get code for all selected plots"""
        if not self.selected_plot_ids:
            return "# No plots selected"
        
        selected_plots = [p for p in self.plot_gallery if p.id in self.selected_plot_ids]
        
        if not selected_plots:
            return "# No plots selected"
        
        code_blocks = []
        for idx, plot in enumerate(selected_plots, 1):
            code_blocks.append(f"# Plot {idx}\n{plot.code}")
        
        separator = "\n\n" + "#" + "="*70 + "\n\n"
        return separator.join(code_blocks)

    def render_gallery(self, num_plots):
        """Render the gallery display with all plots"""
        if not self.plot_gallery:
            empty_updates = [gr.update(visible=False)] * num_plots*2
            return (
                *empty_updates,
                "# No plots generated yet\n\nClick 'Generate Plots' to create visualizations.",
                "Gallery: 0 plots | Selected: 0"
            )

        gallery_updates = []
        for i in range(NUM_PLOTS):
            if i < len(self.plot_gallery):
                plot = self.plot_gallery[i]
                gallery_updates.extend([
                    gr.update(value=plot.figure, visible=True),
                    gr.update(value=plot.selected, visible=True, interactive=True, label=f"Include Plot {i+1}"),  # Checkbox
                ])
            else:
                gallery_updates.extend([
                    gr.update(visible=False),
                    gr.update(visible=False),
                ])

        
        code = self.get_selected_plots_code()
        num_selected = len(self.selected_plot_ids)
        status = f"Gallery: {len(self.plot_gallery)} plot(s) | ✓ Selected: {num_selected}"

        return (*gallery_updates, code, status) 


    def update_selection_display(self):
        """Update only selection-dependent components"""
        code = self.get_selected_plots_code()
        num_selected = len(self.selected_plot_ids)
        status = f"📊 Gallery: {len(self.plot_gallery)} plot(s) | ✓ Selected: {num_selected}"
        
        checkbox_updates = []
        for i in range(NUM_PLOTS):  
            if i < len(self.plot_gallery):
                plot = self.plot_gallery[i]
                checkbox_updates.append(gr.update(value=plot.selected))
            else:
                checkbox_updates.append(gr.update())
        
        return (*checkbox_updates, code, status)

    def _create_error_message(self, message: str):
        """Create error display"""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=11)
        ax.axis('off')
        
        # Return for NUM_PLOTS * 2 components (image + checkbox for each)
        error_updates = [
            gr.update(value=fig if i == 0 else None, visible=i == 0) 
            for i in range(NUM_PLOTS * 2)
        ]
        
        return (
            *error_updates,
            f"# Error\n{message}",
            f"Error: {message}"
        )


    def handle_checkbox_change(self, slot_idx: int, is_checked: bool):
        """
        Handle checkbox state change.
        When a checkbox is clicked, toggle the selection of that plot.
        """
        if slot_idx < len(self.plot_gallery):
            plot = self.plot_gallery[slot_idx]
            if is_checked and plot.id not in self.selected_plot_ids:
                self.selected_plot_ids.add(plot.id)
                plot.selected = True
            elif not is_checked and plot.id in self.selected_plot_ids:
                self.selected_plot_ids.remove(plot.id)
                plot.selected = False
            
            logging.info(f"Plot {slot_idx+1} {'selected' if is_checked else 'deselected'}")
        
        return self.update_selection_display()


    def parse_gradio_dataframe(self, raw_table) -> Optional[pd.DataFrame]:
        """
        Convert gr.Dataframe output to a clean pandas DataFrame.
        """
        if raw_table is None or len(raw_table) == 0:
            return None

        # Remove completely empty rows
        rows = [
            row for row in raw_table
            if any(cell not in ("", None) for cell in row)
        ]

        if len(rows) < 2:
            return None  # need at least header + one row

        header = rows[0]
        if isinstance(header, str):
            header = [header]
        data_rows = rows[1:]

        df = pd.DataFrame(data_rows, columns=header)

        # Attempt numeric conversion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        return df

    # def generate_multiple_plots(self, user_data, num_plots: int):
    #         """
    #         Generates multiple plot variations.
    #         Works in three modes:
    #         - paper-only
    #         - data-only
    #         - paper + data
    #         """

    #         try:
    #             # ---- 1. Parse UI table into DataFrame ----
    #             # df = self.parse_gradio_dataframe(user_data)
    #             has_data = df is not None and not df.empty

    #             # ---- 2. Get draft if available ----
    #             draft = ""
    #             if self.writer is not None:
    #                 state = self.writer.get_state()
    #                 draft = state.values.get("draft", "")

    #             has_draft = bool(draft.strip())

    #             # ---- 3. Validate signal availability ----
    #             if not has_data and not has_draft:
    #                 return self._create_error_message(
    #                     "No input available.\n\n"
    #                     "Please either:\n"
    #                     "• generate paper content, or\n"
    #                     "• enter data in the table to generate plots."
    #                 )

    #             logging.info(
    #                 f"Generating {num_plots} plot(s) | "
    #                 f"draft={'yes' if has_draft else 'no'} | "
    #                 f"data={'yes' if has_data else 'no'}"
    #             )

    #             # ---- 4. Generate plots ----
    #             plot_results = []
    #             for i in range(num_plots):
    #                 fig, code = self.plotter.suggest_plot(
    #                     paper_content=draft if has_draft else "",
    #                     data=df if has_data else None,
    #                     num_plots=num_plots
    #                 )
    #                 plot_results.append((fig, code))

    #             # ---- 5. Store results ----
    #             new_plots = []
    #             for fig, code in plot_results:
    #                 plot_version = PlotVersion(
    #                     id=str(uuid.uuid4()),
    #                     figure=fig,
    #                     code=code,
    #                     timestamp=datetime.now(),
    #                     selected=False
    #                 )
    #                 self.plot_gallery.append(plot_version)
    #                 new_plots.append(plot_version)

    #             logging.info(f"Successfully generated {len(new_plots)} plots.")
    #             return self.render_gallery(num_plots)

    #         except Exception as e:
    #             logging.error("Error in generate_multiple_plots", exc_info=True)
    #             return self._create_error_message(f"Critical Error: {str(e)}")

    def generate_multiple_plots(self, plot_prompt: str, num_plots: int):
        """
        Generates multiple plot variations based on:
        - paper draft (if available)
        - user-provided plot description (plot_prompt)

        Args:
            plot_prompt: str, user's description of the desired plot
            num_plots: int, number of variations to generate
        """

        try:
            # ---- 1. Get paper draft if available ----
            draft = ""
            if self.writer is not None:
                state = self.writer.get_state()
                draft = state.values.get("draft", "")

            has_prompt = bool(plot_prompt.strip())
            has_draft = bool(draft.strip())

            # ---- 2. Validate input availability ----
            if not has_prompt and not has_draft:
                return self._create_error_message(
                    "No input available.\n\n"
                    "Please either:\n"
                    "• generate paper content, or\n"
                    "• enter a plot description to generate plots."
                )

            logging.info(
                f"Generating {num_plots} plot(s) | "
                f"draft={'yes' if has_draft else 'no'} | "
                f"plot_prompt={'yes' if has_prompt else 'no'}"
            )

            # ---- 3. Generate plots ----
            plot_results = []
            for i in range(num_plots):
                fig, code = self.plotter.suggest_plot(
                    paper_content=draft if has_draft else "",
                    user_prompt=plot_prompt if has_prompt else "",
                    num_plots=num_plots
                )
                plot_results.append((fig, code))

            # ---- 4. Store results in gallery ----
            new_plots = []
            for fig, code in plot_results:
                plot_version = PlotVersion(
                    id=str(uuid.uuid4()),
                    figure=fig,
                    code=code,
                    timestamp=datetime.now(),
                    selected=False
                )
                self.plot_gallery.append(plot_version)
                new_plots.append(plot_version)

            logging.info(f"Successfully generated {len(new_plots)} plots.")
            return self.render_gallery(num_plots)

        except Exception as e:
            logging.error("Error in generate_multiple_plots", exc_info=True)
            return self._create_error_message(f"Critical Error: {str(e)}")

    def create_ui(self) -> None:
        with gr.Blocks(
            theme=gr.themes.Default(), fill_height=True
        ) as self.kiroku_agent:
            with gr.Tab("Initial Instructions"), gr.Row():
                file = gr.File(file_types=[".yaml"], scale=1)
                js = gr.JSON(scale=5)

            with gr.Tab("Document Writing"):
                out = gr.Textbox(label="Echo")
                inp = gr.Textbox(placeholder="Instruction", label="Rider")
                markdown = gr.Markdown("")
                doc = gr.Button("Save")

            with gr.Tab("References") as self.ref_block:
                ref_list = [
                    gr.Checkbox(
                        value=False, visible=False, label=False, interactive=True
                    )
                    for _ in range(1000)
                ]
                submit_ref_list = gr.Button("Submit", visible=False)

            plot_components = []
            plot_imgs = []
            plot_checkboxes = []
            with gr.Tab("Plot Suggestion") as self.plottab:
                gr.Markdown("""# 📊 Plot Generation""")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📋 Data Input (Optional)")
                        gr.Markdown("""
                        Paste data here or leave empty to generate plots based on paper content.

                        **Formats accepted:**
                        - CSV data (copy-paste from Excel/CSV)
                        - Manual entry in table
                        - Leave empty to use synthetic data from paper context
                        """)

                        data_input = gr.Textbox(
                            label="Plot description",
                            placeholder="e.g. Show the distribution of response times across conditions",
                            lines=3
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Generation Settings")

                        num_plots_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            step=1,
                            label="Number of plot variations",
                            info="Generate multiple distinct visualizations"
                        )

                        generate_btn = gr.Button(
                            "🎨 Generate Plots",
                            variant="primary",
                            size="lg"
                        )


                gr.Markdown("---")
                gr.Markdown("### 🖼️ Generated Plots")

                with gr.Row():
                    for i in range(NUM_PLOTS):
                        with gr.Column():
                            with gr.Group():
                                plot_img = gr.Plot(label=f"Plot {i+1}", visible=False)
                                plot_checkbox = gr.Checkbox(
                                    label=f"✓ Select for paper",
                                    value=False,
                                    visible=False,
                                    interactive=True
                                )

                                plot_imgs.append(plot_img)
                                plot_checkboxes.append(plot_checkbox)
                                plot_components.extend([plot_img, plot_checkbox])

                gr.Markdown("### 📝 Code for Selected Plots")
                selected_code = gr.Code(
                    label="Python code",
                    language="python",
                    lines=20,
                    value="# Select plots above to view their code"
                )

                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to generate plots",
                    interactive=False
                )

                # Wire up the generate button
                generate_btn.click(
                    self.generate_multiple_plots,
                    inputs=[data_input, num_plots_slider],
                    outputs=[*plot_components, selected_code, status_text]
                )

                # Wire up checkboxes
                for idx, checkbox in enumerate(plot_checkboxes):
                    checkbox.change(
                        lambda is_checked, slot_idx=idx: self.handle_checkbox_change(
                            slot_idx, is_checked
                        ),
                        inputs=[checkbox],
                        outputs=[*plot_checkboxes, selected_code, status_text]
                    )



            inp.submit(self.update, inp, [markdown, out, inp]).then(
                lambda: gr.update(
                    value="",
                    interactive=self.next_state
                    not in [END, "generate_citations", "NONE"],
                ),
                [],
                inp,
            ).then(self.update_refs, [], [submit_ref_list, *ref_list])
            file.upload(self.process_file, file, [js, inp]).then(
                self.initial_step, [], [markdown, out]
            ).then(lambda: gr.update(placeholder="", interactive=True), [], inp)
            doc.click(self.save_as, [], out)
            submit_ref_list.click(
                self.submit_ref_list, ref_list, [markdown, out, submit_ref_list]
            )

    def launch_ui(self):
        logging.warning(
            f"... using KIROKU_PROJECT_DIRECTORY working directory of {self.working_dir}"
        )
        try:
            os.chdir(self.working_dir)
        except Exception as err:
            logging.warning(f"... directory {self.working_dir} does not exist, {err}")
            self.working_dir.mkdir(exist_ok=True, parents=True)
        self.images = self.working_dir / "images"
        logging.warning(
            f"... using directory {self.working_dir}/images to store images"
        )
        try:
            self.images.mkdir(exist_ok=True, parents=True)
        except Exception as err:
            logging.error(f"Error occurred while creating images directory, {err}")
        self.kiroku_agent.launch(
            server_name="localhost"
        )

def run() -> None:
    working_dir: Path = config(
        "KIROKU_PROJECT_DIRECTORY", default=Path.cwd(), cast=Path
    )
    # need this to allow images to be in a different directory
    gr.set_static_paths(paths=[working_dir / "images"])
    kiroku = KirokuUI(working_dir)
    kiroku.create_ui()
    kiroku.launch_ui()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    run()
