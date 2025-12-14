# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gradio as gr
import markdown
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from gradio.components.html import HTML
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.states import *
from src.agents.suggest_plot_base import PlotSuggester
from src.utils.models import PaperConfig

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from decouple import config

from src.utils.constants import SUPPORTED_TABLE_FORMATS
from src.agents.prompts import VARIED_PLOT_PROMPT

PREINITIALIZED_COMPONENTS = 30
NUM_PLOTS = 3


@dataclass
class PlotVersion:
    """Represents a generated plot version"""

    id: str
    image_path: str | Path
    code: str
    timestamp: datetime
    selected: bool = False


class DocumentWriter:
    NODE_SUFFIX = "_graph_state"

    def __init__(
        self,
        suggest_title: bool = False,
        generate_citations: bool = True,
        model_name: str = "openai",
        temperature: float = 0.0,
        relevant_files: list[RelevantFile] | None = None,
    ) -> None:
        if relevant_files is None:
            relevant_files = []
        self.suggest_title = suggest_title
        self.generate_citations = generate_citations
        self.state = None
        self.set_thread_id(1)

        # TODO make models configurable for different tasks
        self.model_m = ChatOpenAI(
            model=model_name, temperature=temperature, api_key=config("OPENAI_API_KEY")
        )
        self.state_nodes = {
            node.name: node
            for node in [
                AnalyzeRelevantFiles(self.model_m),
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
        base_name = name.removesuffix(self.NODE_SUFFIX)
        if (
            base_name in ["suggest_title", "suggest_title_review"]
            and not self.suggest_title
        ):
            return False
        return not (
            base_name in ["generate_references", "generate_citations"]
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
                "suggest_title_review_graph_state",
                self.is_title_review_complete,
                {
                    "next_phase": "internet_search_graph_state",
                    "review_more": "suggest_title_graph_state",
                },
            )
        builder.add_conditional_edges(
            "topic_sentence_manual_review_graph_state",
            self.is_plan_review_complete,
            {
                "topic_sentence_manual_review_graph_state": "topic_sentence_manual_review_graph_state",
                "paper_writer_graph_state": "paper_writer_graph_state",
            },
        )
        builder.add_conditional_edges(
            "writer_manual_reviewer_graph_state",
            self.is_generate_review_complete,
            {
                "writer_manual_reviewer_graph_state": "writer_manual_reviewer_graph_state",
                "reflection_reviewer_graph_state": "reflection_reviewer_graph_state",
                "write_abstract_graph_state": "write_abstract_graph_state",
            },
        )
        if self.suggest_title:
            builder.add_edge(
                "analyze_relevant_files_graph_state", "suggest_title_graph_state"
            )
            builder.add_edge(
                "suggest_title_graph_state", "suggest_title_review_graph_state"
            )
        else:
            builder.add_edge(
                "analyze_relevant_files_graph_state", "internet_search_graph_state"
            )
        builder.add_edge(
            "internet_search_graph_state", "topic_sentence_writer_graph_state"
        )
        builder.add_edge(
            "topic_sentence_writer_graph_state",
            "topic_sentence_manual_review_graph_state",
        )
        builder.add_edge(
            "paper_writer_graph_state", "writer_manual_reviewer_graph_state"
        )
        builder.add_edge(
            "reflection_reviewer_graph_state",
            "additional_reflection_instructions_graph_state",
        )
        builder.add_edge(
            "additional_reflection_instructions_graph_state", "paper_writer_graph_state"
        )
        if self.generate_citations:
            builder.add_edge(
                "write_abstract_graph_state", "generate_references_graph_state"
            )
            builder.add_edge(
                "generate_references_graph_state", "generate_citations_graph_state"
            )
            builder.add_edge(
                "generate_citations_graph_state", "generate_figure_captions_graph_state"
            )
        else:
            builder.add_edge(
                "write_abstract_graph_state", "generate_figure_captions_graph_state"
            )
        builder.add_edge(
            "generate_figure_captions_graph_state", "latex_converter_graph_state"
        )
        builder.add_edge("latex_converter_graph_state", END)

        builder.set_entry_point("analyze_relevant_files_graph_state")

        self.interrupt_after = []
        self.interrupt_before = (
            ["suggest_title_review_graph_state"] if self.suggest_title else []
        )
        self.interrupt_before.extend(
            [
                "topic_sentence_manual_review_graph_state",
                "writer_manual_reviewer_graph_state",
                "additional_reflection_instructions_graph_state",
            ]
        )
        if self.generate_citations:
            self.interrupt_before.append("generate_citations_graph_state")
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

        if not state.messages:
            return "next_phase"
        return "review_more"

    def is_plan_review_complete(self, state: AgentState, config: dict) -> str:
        """
        Checks if plan manual review is complete based on an empty instruction.

        :param state: state of agent.
        :return: next state of agent.
        """
        if config["configurable"]["instruction"]:
            return "topic_sentence_manual_review_graph_state"
        return "paper_writer_graph_state"

    def is_generate_review_complete(self, state: AgentState, config: dict) -> str:
        """
        Checks if review of generation phase is complete based on number of revisions.

        :param state: state of agent.
        :return: next state to go.
        """
        if config["configurable"]["instruction"]:
            return "writer_manual_reviewer_graph_state"
        if state.revision_number <= state.max_revisions:
            return "reflection_reviewer_graph_state"
        return "write_abstract_graph_state"

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

    def stream_generator(self, state: PaperConfig, config: dict):
        """
        Invokes the multi-agent system to write a paper.

        :param state: state of initial invokation.
        :return: full state information
        """
        config = {"configurable": config}
        config["configurable"]["thread_id"] = self.get_thread_id()
        for event in self.graph.stream(state, config, stream_mode="values"):
            self.state = event
            yield event

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
        try:
            img = self.graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API
            )
            display(Image(img))

            with open("kiroku_graph.png", "wb") as f:
                f.write(img)
        except Exception as e:
            logging.warning(f"Could not draw graph: {e}")


class KirokuUI:
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.first = True
        self.next_state = -1
        self.references = []
        self.state_values = PaperConfig()
        self.writer = None

        self.plotter = None

        # Log rendering caches
        self._last_rendered_log_len: int = 0
        self._cached_steps_order: list[str] = []
        self._cached_markdown_by_step: dict[str, str] = {}
        self._last_accordion_updates: list = [gr.update(visible=False)] * PREINITIALIZED_COMPONENTS
        self._last_markdown_updates: list = [gr.update(value="")] * PREINITIALIZED_COMPONENTS

    def _initialize_plotter(self):
        """Initialize PlotSuggester"""
        try:
            model = ChatOpenAI(
                model=self.state_values.model_name,
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
        """
        Reads initial state from a YAML 'filename'.
        :param filename: YAML file containing initial paper configuration.
        :return: initial state dictionary.
        """
        with filename.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logging.info(f"Initial yaml config: {data}")

        relevant_files = self._get_relevant_files(data)

        cfg = PaperConfig(**data, relevant_files=relevant_files)
        cfg.hypothesis = "\n\n".join(filter(None, [cfg.hypothesis, cfg.instructions]))

        return cfg

    def step(self, instruction: str, state_values: PaperConfig = None):
        """
        Performs one step of the graph invocation, stopping at the next break point.
        :param instruction: instruction to execute.
        :param state_values: initial state values or None if continuing.
        :return: draft of the paper.
        """
        config = {"instruction": instruction}
        for event in self.writer.stream_generator(state_values, config):
            draft = event.get("draft", "").strip()
            # we have to do this because the LLM sometimes decide to add
            # this to the final answer.
            if "```markdown" in draft:
                draft = "\n".join(draft.split("\n")[1:-1])
            yield draft

    def _get_log_updates(self):
        if not self.writer:
            # cache empty updates
            self._last_accordion_updates = [gr.update(visible=False)] * PREINITIALIZED_COMPONENTS
            self._last_markdown_updates = [gr.update(value="")] * PREINITIALIZED_COMPONENTS
            return self._last_accordion_updates, self._last_markdown_updates

        state = self.writer.get_state()
        logs = state.values.get("workflow_logs", [])

        # If no new logs since last render, reuse cached UI
        if len(logs) == self._last_rendered_log_len:
            return self._last_accordion_updates, self._last_markdown_updates

        # Process only new logs and append to cache structures
        new_logs = logs[self._last_rendered_log_len :]
        for log in new_logs:
            step = log.step
            if step not in self._cached_markdown_by_step:
                self._cached_markdown_by_step[step] = ""
                self._cached_steps_order.append(step)
            # Append formatted entry incrementally
            self._cached_markdown_by_step[step] += f"### {log.side}\n{log.message}\n\n---\n\n"

        # Update cursor
        self._last_rendered_log_len = len(logs)

        # Build UI updates for up to PREINITIALIZED_COMPONENTS steps
        accordion_updates = []
        markdown_updates = []
        for i in range(PREINITIALIZED_COMPONENTS):
            if i < len(self._cached_steps_order):
                step_name = self._cached_steps_order[i]
                content = self._cached_markdown_by_step.get(step_name, "")
                accordion_updates.append(gr.update(visible=True, label=step_name))
                markdown_updates.append(gr.update(value=content))
            else:
                accordion_updates.append(gr.update(visible=False))
                markdown_updates.append(gr.update(value=""))

        # Cache latest UI updates for reuse during streaming
        self._last_accordion_updates = accordion_updates
        self._last_markdown_updates = markdown_updates
        return accordion_updates, markdown_updates

    def update(self, instruction):
        """
        Updates state upon submitting an instruction or updating references.
        :param instruction: instruction to be executed.
        :return: new draft, atlas message and making input object non-interactive.
        """
        draft = ""
        # STREAM PHASE: only update draft and state, reuse cached log UI
        for d in self.step(instruction):
            draft = d

            current_state_val = "Processing..."
            if self.writer and self.writer.state:
                current_state_val = self.writer.state.get("state", "Unknown")

            yield (
                draft,
                current_state_val,
                "Processing...",
                gr.update(interactive=False),
                *self._last_accordion_updates,
                *self._last_markdown_updates,
            )

        # FINAL PHASE: compute logs/references and emit full UI (3.2)
        state = self.writer.get_state()
        current_state = state.values["state"]
        try:
            next_state = state.next[0]
        except:
            next_state = "NONE"

        # if state is in reflection stage, draft to be shown is in the critique field.
        if (
            current_state == "reflection_reviewer_graph_state"
            and next_state == "additional_reflection_instructions_graph_state"
        ):
            draft = state.values["critique"]

        # if next state is going to generate citations, we populate the references
        # for the Tab References.
        if next_state == "generate_citations_graph_state":
            self.references = state.values.get("references", []).split("\n")

        # if we have reached the end, we will save everything.
        if next_state == END or next_state == "NONE":
            dir = os.path.splitext(self.filename)[0]
            logging.warning(f"saving final draft in {dir}")
            self.save_as()

        self.next_state = next_state

        # Refresh log UI once at the end of stream
        log_accordions, log_markdowns = self._get_log_updates()

        yield (
            draft,
            current_state,
            self.atlas_message(next_state),
            gr.update(interactive=False),
            *log_accordions,
            *log_markdowns,
        )

    def atlas_message(self, state):
        """
        Returns the Echo message for a given state.
        :param state: Next state of the multi-agent system.
        :return:
        """
        message = {
            "suggest_title_review_graph_state": "Please suggest review instructions for the title.",
            "topic_sentence_manual_review_graph_state": "Please suggest review instructions for the topic sentences.",
            "writer_manual_reviewer_graph_state": "Please suggest review instructions for the main draft.",
            "additional_reflection_instructions_graph_state": "Please provide additional instructions for the overall paper review.",
            "generate_citations_graph_state": "Please look at the references tab and confirm the references.",
        }

        instruction = message.get(state, "")
        if instruction or state == "generate_citations_graph_state":
            if state == "generate_citations_graph_state":
                return instruction
            return instruction + " Type <RETURN> when done."
        return "We have reached the end."

    def initial_step(self):
        """
        Performs initial step, in which we need to providate a state to the graph.
        :return: draft and Echo message.
        """
        state_values = self.state_values.model_copy(deep=True)
        if self.state_values.suggest_title:
            state_values.state = "suggest_title_graph_state"
        else:
            state_values.state = "topic_sentence_writer_graph_state"

        draft = ""
        # STREAM PHASE: only draft/state, reuse cached logs
        for d in self.step("", state_values):
            draft = d

            current_state_val = "Processing..."
            if self.writer and self.writer.state:
                current_state_val = self.writer.state.get("state", "Unknown")

            yield (
                draft,
                current_state_val,
                "Processing...",
                gr.update(value="", interactive=False),
                *self._last_accordion_updates,
                *self._last_markdown_updates,
            )

        state = self.writer.get_state()
        current_state = state.values["state"]
        try:
            next_state = state.next[0]
        except:
            next_state = "NONE"

        # Refresh log UI once at the end of initial stream
        log_accordions, log_markdowns = self._get_log_updates()

        yield (
            draft,
            current_state,
            self.atlas_message(next_state),
            gr.update(
                value="",
                interactive=next_state not in [END, "generate_citations", "NONE"],
            ),
            *log_accordions,
            *log_markdowns,
        )

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

        self.working_dir = self.state_values.working_dir

        # Reset log caches on new document
        self._last_rendered_log_len = 0
        self._cached_steps_order.clear()
        self._cached_markdown_by_step.clear()
        self._last_accordion_updates = [gr.update(visible=False)] * PREINITIALIZED_COMPONENTS
        self._last_markdown_updates = [gr.update(value="")] * PREINITIALIZED_COMPONENTS

        relevant_files_preview = self._update_files_preview()

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

        return (
            *relevant_files_preview,
            self.state_values.model_dump_json(),
            gr.update(interactive=False),
        )

    def save_as(self):
        """
        Saves project status. We save all instructions given by the user.
        :return: message where the project was saved.
        """
        filename = self.filename
        state = self.writer.get_state()

        draft = state.values.get("draft", "")
        latex_draft = state.values.get("latex_draft", "")
        # need to replace file= by empty because of gradio problem in Markdown
        draft = re.sub(r"\/?file=", "", draft)

        # Convert relative image paths to absolute paths for Gradio compatibility
        # This helps images display during generation process
        plan = state.values.get("plan", "")
        review_topic_sentences = "\n\n".join(
            state.values.get("review_topic_sentences", [])
        )
        review_instructions = "\n\n".join(state.values.get("review_instructions", []))
        content = "\n\n".join(state.values.get("content", []))

        dir_ = filename.parent / filename.stem

        dir_.mkdir(exist_ok=True, parents=True)

        # Don't create symlinks on Windows - use absolute paths instead
        # Images are already handled with absolute paths in the content
        # No need for symlinks since we're using absolute paths in markdown
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

        except:
            logging.error("cannot find 'pandoc'")

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
        ] + [gr.update()] * (200 - len(self.references))
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
        :return: Everything read by self.update.
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
        yield from self.update("")

    def create_ui(self) -> None:
        with gr.Blocks(
            theme=gr.themes.Default(), fill_height=True, css=".center-col{display:flex; justify-content:center;}"
        ) as self.kiroku_agent:
            with gr.Tab("Initial Instructions"):
                with gr.Row():
                    file = gr.File(file_types=[".yaml"], scale=1)
                    js = gr.JSON(scale=5)
                _, _, files_preview = self._make_files_preview()
            with gr.Tab("Document Writing"):
                current_state = gr.Textbox(label="Current State")
                out = gr.Textbox(label="Echo")
                inp = gr.Textbox(placeholder="Instruction", label="Rider")
                markdown = gr.Markdown("")

            with gr.Tab("References") as self.ref_block:
                ref_list = [
                    gr.Checkbox(
                        value=False, visible=False, label=False, interactive=True
                    )
                    for _ in range(200)
                ]
                submit_ref_list = gr.Button("Submit", visible=False)

            with gr.Tab("Instruction log"):
                _, self.log_accordions, self.log_markdowns = (
                    self._logs_viewer_elements()
                )

            with gr.Tab("🎨 Plot Generation"):
                gr.Markdown("# Plot Generator")

                with gr.Row():
                    with gr.Column(scale=2):
                        plot_prompt = gr.Textbox(
                            label="Plot Description",
                            placeholder="e.g., Create a sine wave with random noise overlay",
                            lines=3,
                        )

                    with gr.Column(scale=1, elem_classes=["center-col"]):
                        generate_btn = gr.Button(
                            "✨ Generate", variant="primary", size="lg"
                        )

                status_text = gr.Textbox(
                    label="Status", value="Ready", interactive=False
                )

                gr.Markdown("### Generated Plots")

                # Plot gallery (just images, no checkboxes)
                plot_images = []

                with gr.Row():
                    for i in range(NUM_PLOTS):
                        with gr.Column():
                            plot_img = gr.Image(
                                label=f"Variation {i + 1}", visible=False, type="filepath"
                            )
                            plot_images.append(plot_img)

                # Radio button for selection
                gr.Markdown("### Select Plot to View Code")
                plot_selector = gr.Radio(
                    choices=[],
                    label="Choose a plot",
                    value=None,
                    visible=False,
                    interactive=True,
                )

                gr.Markdown("### Code for Selected Plot")
                selected_code = gr.Code(
                    label="Python Code",
                    language="python",
                    lines=15,
                    value="# Generate plots above to view code",
                )

            generate_btn.click(
                self.generate_multiple_plots,
                inputs=[plot_prompt],
                outputs=[*plot_images, plot_selector, selected_code, status_text],
                concurrency_limit=10,
                concurrency_id="plot_generation",
            )

            # Radio button selection
            plot_selector.change(
                fn=self.show_single_plot_code,
                inputs=[plot_selector],
                outputs=[selected_code, status_text],
            )

            inp.submit(
                self.update,
                inp,
                [
                    markdown,
                    current_state,
                    out,
                    inp,
                    *self.log_accordions,
                    *self.log_markdowns,
                ],
                concurrency_limit=10,
                concurrency_id="main_flow",
            ).then(
                lambda: gr.update(
                    value="",
                    interactive=self.next_state
                    not in [END, "generate_citations", "NONE"],
                ),
                [],
                inp,
            ).then(self.update_refs, [], [submit_ref_list, *ref_list])
            file.upload(self.process_file, file, [*files_preview, js, inp]).then(
                self.initial_step,
                [],
                [
                    markdown,
                    current_state,
                    out,
                    inp,
                    *self.log_accordions,
                    *self.log_markdowns,
                ],
                concurrency_limit=10,
                concurrency_id="main_flow",
            ).then(lambda: gr.update(placeholder="", interactive=True), [], inp)

            # doc.click(self.save_as, [], out)
            submit_ref_list.click(
                self.submit_ref_list,
                ref_list,
                [
                    markdown,
                    current_state,
                    out,
                    inp,
                    *self.log_accordions,
                    *self.log_markdowns,
                ],
            )

    # def generate_multiple_plots(self, plot_prompt: str):
    #     """Generate multiple plot variations using fixed NUM_PLOTS"""

    #     try:
    #         if not self.plotter:
    #             self.plotter = self._initialize_plotter()

    #         if not self.plotter:
    #             yield self._create_error_message(
    #                 "Could not initialize plotter. Please check logs."
    #             )
    #             return

    #         # Clear previous gallery
    #         self.plot_gallery = []
    #         self.selected_plot_id = None

    #         # Get paper context if available
    #         paper_context = ""
    #         relevant_files = []

    #         # Try to get latest context from writer if available
    #         if self.writer:
    #             try:
    #                 state = self.writer.get_state()
    #                 if state and state.values:
    #                     if state.values.get("relevant_files"):
    #                         relevant_files = state.values["relevant_files"]

    #                     else:
    #                         yield self._create_error_message(
    #                             "⚠️ No relevant files found in writer state."
    #                         )
    #                         return
    #                     if state.values.get("draft"):
    #                         paper_context = state.values["draft"]
    #                     elif state.values.get("content"):
    #                         content = state.values["content"]
    #                         paper_context = (
    #                             "\n\n".join(content)
    #                             if isinstance(content, list)
    #                             else str(content)
    #                         )
    #                     elif state.values.get("hypothesis"):
    #                         paper_context = state.values["hypothesis"]
    #             except Exception as e:
    #                 logging.warning(f"Could not retrieve writer state: {e}")

    #         if (
    #             not paper_context
    #             and self.state_values
    #             and hasattr(self.state_values, "hypothesis")
    #         ):
    #             paper_context = self.state_values.hypothesis

    #         has_prompt = bool(plot_prompt.strip())
    #         has_draft = bool(paper_context.strip())

    #         if not has_prompt and not has_draft and not relevant_files:
    #             yield self._create_error_message(
    #                 "⚠️ No input provided\n\nPlease provide a plot description."
    #             )
    #             return

    #         # Generate plots with VARIATION in the prompt
    #         for relevant_file in relevant_files:
    #             logging.info(
    #                 f"Generating plot for relevant file: {relevant_file.file_path}"
    #             )
    #             for i in range(NUM_PLOTS):
    #                 logging.info(f"Generating variation {i + 1}/{NUM_PLOTS}...")

    #                 # ⭐ Show "generating" status immediately
    #                 status_update = f"🔄 Generating plot {i + 1}/{NUM_PLOTS}..."
    #                 current_gallery = self.render_gallery()
    #                 # Update just the status text (last element in the tuple)
    #                 yield (*current_gallery[:-1], status_update)

    #                 try:
    #                     # Add variation instruction to prompt
    #                     varied_prompt = f"{plot_prompt}\n\nThis is variation {i+1} of {NUM_PLOTS}. Make it distinct."

    #                     fig_path, code = self.plotter.suggest_plot(
    #                         relevant_file=relevant_file,
    #                         paper_content=paper_context if has_draft else "",
    #                         user_prompt=varied_prompt,  # ← Use varied prompt
    #                         num_plots=NUM_PLOTS,
    #                     )

    #                     plot_version = PlotVersion(
    #                         id=str(uuid.uuid4()),
    #                         image_path=fig_path,
    #                         code=code,  # ← Each should now have different code
    #                         timestamp=datetime.now(),
    #                         selected=False,
    #                     )

    #                     self.plot_gallery.append(plot_version)
    #                     logging.info(f"✓ Variation {i + 1} generated")
    #                     yield self.render_gallery()

    #                     # Log code preview to verify it's different
    #                     logging.debug(
    #                         f"Variation {i + 1} code preview: {code[:100]}..."
    #                     )

    #                 except Exception as e:
    #                     logging.error(f"Failed to generate variation {i + 1}: {e}")
    #                     continue

    #             success_count = len(self.plot_gallery)
    #             logging.info(f"✓ Complete: {success_count}/{NUM_PLOTS} successful")

    #             # yield self.render_gallery()

    #     except Exception as e:
    #         logging.error("Critical error in generate_multiple_plots", exc_info=True)
    #         yield self._create_error_message(f"Critical Error:\n{str(e)}")
    #     finally:
    #         if self.plotter:
    #             self.plotter.cleanup()
    def generate_multiple_plots(self, plot_prompt: str):
        """Generate one plot per relevant file"""

        try:
            # Initialize plotter
            if not self.plotter:
                self.plotter = self._initialize_plotter()
            if not self.plotter:
                yield self._create_error_message("Could not initialize plotter. Please check logs.")
                return

            # Clear previous plots
            self.plot_gallery = []
            self.selected_plot_id = None

            # Validate input
            if not plot_prompt.strip():
                yield self._create_error_message("No plot description provided.")
                return

            # Get context from writer state
            paper_context, relevant_files = self._get_writer_context()

            if not relevant_files:
                yield self._create_error_message("No relevant files found.")
                return

            # Generate one plot per file
            total_files = len(relevant_files)

            for i, relevant_file in enumerate(relevant_files, 1):
                # Show progress status
                status = f"🔄 Generating plot {i}/{total_files} for {relevant_file.file_path.name}..."
                yield (*self.render_gallery()[:-1], status)

                try:
                    # Generate plot
                    fig_path, code = self.plotter.suggest_plot(
                        relevant_file=relevant_file,
                        paper_content=paper_context,
                        user_prompt=plot_prompt,
                        # num_plots=1,
                    )

                    # Add to gallery
                    self.plot_gallery.append(PlotVersion(
                        id=str(uuid.uuid4()),
                        image_path=fig_path,
                        code=code,
                        timestamp=datetime.now(),
                        selected=False,
                    ))

                    logging.info(f"✓ Generated plot {i}/{total_files}")

                    # Update UI with new plot
                    yield self.render_gallery()

                except Exception as e:
                    logging.error(f"Failed to generate plot for {relevant_file.file_path.name}: {e}")
                    continue

            # Final summary
            success_count = len(self.plot_gallery)
            logging.info(f"✓ Complete: {success_count}/{total_files} plots generated")

        except Exception as e:
            logging.error("Critical error in generate_multiple_plots", exc_info=True)
            yield self._create_error_message(f"Critical Error:\n{str(e)}")
        finally:
            if self.plotter:
                self.plotter.cleanup()


    def _get_writer_context(self):
        """Extract paper context and relevant files from writer state"""
        paper_context = ""
        relevant_files = []

        if self.writer:
            try:
                state = self.writer.get_state()
                if not state or not state.values:
                    logging.warning("Writer state not ready yet")
                    return paper_context, relevant_files

                if state and state.values:
                    # Get relevant files
                    relevant_files = state.values.get("relevant_files", [])

                    if state.values.get("draft"):
                        paper_context = state.values["draft"]
                    elif state.values.get("content"):
                        content = state.values["content"]
                        paper_context = "\n\n".join(content) if isinstance(content, list) else str(content)
                    elif state.values.get("hypothesis"):
                        paper_context = state.values["hypothesis"]
            except Exception as e:
                logging.warning(f"Could not retrieve writer state: {e}")

        # Fallback to state_values
        if not paper_context and self.state_values and hasattr(self.state_values, "hypothesis"):
            paper_context = self.state_values.hypothesis

        return paper_context, relevant_files


    def render_gallery(self):
        """Render the plot gallery with radio choices"""
        if not self.plot_gallery:
            empty_updates = [gr.update(visible=False)] * NUM_PLOTS
            return (
                *empty_updates,
                gr.update(choices=[], value=None),  # Radio button
                "# No plots generated",
                "Ready",
            )

        # Update plot displays
        plot_updates = []
        for i in range(NUM_PLOTS):
            if i < len(self.plot_gallery):
                plot = self.plot_gallery[i]
                plot_updates.append(gr.update(value=str(plot.image_path), visible=True))
            else:
                plot_updates.append(gr.update(visible=False))

        # Update radio button choices
        radio_choices = [f"Plot {i + 1}" for i in range(len(self.plot_gallery))]
        radio_update = gr.update(
            choices=radio_choices,
            value=radio_choices[0] if radio_choices else None,
            visible=True,
        )

        # Show first plot's code by default
        code = self.plot_gallery[0].code if self.plot_gallery else "# No code"
        status = f"✓ {len(self.plot_gallery)} plots generated"

        return (*plot_updates, radio_update, code, status)

    def show_single_plot_code(self, plot_choice: str):
        """Show code for selected plot from radio button"""
        if not plot_choice or not self.plot_gallery:
            return "# No plot selected", gr.update()  # Don't change status

        try:
            # Parse "Plot 1", "Plot 2", etc.
            plot_idx = int(plot_choice.split()[-1]) - 1

            if 0 <= plot_idx < len(self.plot_gallery):
                self.selected_plot_id = self.plot_gallery[plot_idx].id
                code = self.plot_gallery[plot_idx].code
                # Only update status, don't overwrite it completely
                status = f"Viewing: Plot {plot_idx + 1}"
                return code, status
        except Exception as e:
            logging.error(f"Error showing plot code: {e}")

        return "# Error loading plot code", "Error"

    def _create_error_message(self, message: str):
        """Create error display"""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
        ax.axis("off")

        # Convert figure to image array for gr.Image
        fig.canvas.draw()
        try:
            # Use buffer_rgba to avoid backend-specific tostring_rgb issues
            buf = np.asarray(fig.canvas.buffer_rgba())
            # Drop alpha channel for gr.Image if needed
            image_array = buf[:, :, :3]
        except Exception:
            # Fallback: render to PNG bytes and let gr.Image accept filepaths/arrays elsewhere
            from io import BytesIO
            png_buf = BytesIO()
            fig.savefig(png_buf, format="png")
            png_buf.seek(0)
            import PIL.Image as PILImage
            pil_img = PILImage.open(png_buf).convert("RGB")
            image_array = np.array(pil_img)
        plt.close(fig)

        error_updates = []
        for i in range(NUM_PLOTS):
            if i == 0:
                error_updates.append(gr.update(value=image_array, visible=True))
            else:
                error_updates.append(gr.update(visible=False))

        return (
            *error_updates,
            gr.update(choices=[], value=None),  # Radio
            f"# Error\n{message}",
            f"{message}",
        )

    def launch_ui(self):
        logging.warning(
            f"... using KIROKU_PROJECT_DIRECTORY working directory of {self.working_dir}"
        )
        try:
            os.chdir(self.working_dir)
        except Exception as err:
            logging.warning(f"... directory {self.working_dir} does not exist, {err}")

        self.images = self.working_dir / "images"
        self.images.mkdir(parents=True, exist_ok=True)

        # Create and launch the UI
        self.create_ui()
        self.kiroku_agent.queue(default_concurrency_limit=10).launch()

    def _get_relevant_files(self, state_values: dict) -> list[RelevantFile]:
        working_dir = Path(state_values["working_dir"])

        if not working_dir.exists() or not working_dir.is_dir():
            return []

        file_names_and_descs = {
            rel_file["file_name"]: rel_file.get("description", "")
            for rel_file in state_values["files_descriptions"]
        }

        relevant_files = []
        for file in working_dir.rglob("*"):
            for file_name, description in file_names_and_descs.items():
                if (
                    file.suffix
                    in [
                        ".yaml",
                        ".yml",
                        ".md",
                        ".tex",
                        ".html",
                        *SUPPORTED_TABLE_FORMATS,
                    ]
                    and file.name == file_name
                ):
                    relevant_files.append(
                        RelevantFile(file_path=file, description=description)
                    )

        return relevant_files

    def _logs_viewer_elements(self, n_slots: int = PREINITIALIZED_COMPONENTS):
        accordions = []
        markdowns = []

        with gr.Column() as col:
            for i in range(n_slots):
                with gr.Accordion(label=f"Step {i}", open=False, visible=False) as acc:
                    md = gr.Markdown()
                    accordions.append(acc)
                    markdowns.append(md)

        return col, accordions, markdowns

    def _make_files_preview(
        self, n_slots: int = PREINITIALIZED_COMPONENTS
    ) -> tuple[gr.Column, dict, list]:
        file_blocks = {"yaml": [], "markdown": [], "html": [], "table": [], "other": []}

        with gr.Column() as col:
            for i in range(n_slots):
                file_blocks["yaml"].append(
                    gr.Code(visible=False, label=f"yaml_{i}", language="yaml")
                )
                file_blocks["markdown"].append(
                    gr.Markdown(visible=False, label=f"md_{i}")
                )
                file_blocks["html"].append(gr.HTML(visible=False, label=f"html_{i}"))
                file_blocks["table"].append(
                    gr.Dataframe(visible=False, label=f"table_{i}")
                )
                file_blocks["other"].append(
                    gr.Markdown(visible=False, label=f"other_{i}")
                )

        all_components = [c for lst in file_blocks.values() for c in lst]
        return col, file_blocks, all_components

    def _update_files_preview(self) -> list:
        relevant_files = self.state_values.relevant_files
        updates = []

        def pad(updates_list: list, n_left: int) -> list:
            return updates_list + [
                gr.update(visible=False) for _ in range(n_left - len(updates_list))
            ]

        yaml_updates, md_updates, html_updates, table_updates, other_updates = (
            [],
            [],
            [],
            [],
            [],
        )

        for rel_file in relevant_files:
            file = rel_file.file_path
            match file.suffix:
                case ".yaml" | ".yml":
                    yaml_updates.append(
                        gr.update(value=file.read_text(), visible=True, label=file.name)
                    )
                case ".md" | ".tex":
                    md_updates.append(
                        gr.update(value=file.read_text(), visible=True, label=file.name)
                    )
                case ".html":
                    html_updates.append(
                        gr.update(value=file.read_text(), visible=True, label=file.name)
                    )
                case ".csv":
                    table = pl.scan_csv(file).head(5).collect().to_pandas()
                    table_updates.append(
                        gr.update(value=table, visible=True, label=file.name)
                    )
                case ".parquet":
                    table = pl.scan_parquet(file).head(5).collect().to_pandas()
                    table_updates.append(
                        gr.update(value=table, visible=True, label=file.name)
                    )
                case _:
                    other_updates.append(
                        gr.update(
                            value=f"Unsupported file type: {file}",
                            visible=True,
                            label=file.name,
                        )
                    )

        updates.extend(pad(yaml_updates, PREINITIALIZED_COMPONENTS))
        updates.extend(pad(md_updates, PREINITIALIZED_COMPONENTS))
        updates.extend(pad(html_updates, PREINITIALIZED_COMPONENTS))
        updates.extend(pad(table_updates, PREINITIALIZED_COMPONENTS))
        updates.extend(pad(other_updates, PREINITIALIZED_COMPONENTS))

        return updates


def run():
    """Main entry point for the application."""

    # Get working directory from environment variable or use current directory
    working_dir = Path(os.environ.get("KIROKU_PROJECT_DIRECTORY", "."))

    # Create and launch the UI
    ui = KirokuUI(working_dir)
    ui.launch_ui()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    run()
