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

from src.agents.states import *
from src.agents.suggest_plot import *
from src.agents.generate_plot import *
from src.utils.models import PaperConfig
from src.agents.suggest_plot import PlotSuggester

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from decouple import config

from src.utils.constants import SUPPORTED_TABLE_FORMATS

PREINITIALIZED_COMPONENTS = 100
NUM_PLOTS=5

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


class DocumentWriter:
    NODE_SUFFIX = "_graph_state"

    def __init__(
        self,
        suggest_title: bool = False,
        generate_citations: bool = True,
        model_name: str = "openai",
        temperature: float = 1.0,
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
                # PlotSuggestionAgent(self.model_m),
                # PlotApprovalAgent(self.model_m),
                PlotGenerationAgent(self.model_m),
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
                "plot_suggestion_graph_state": "plot_suggestion_graph_state",
            },
        )

        builder.add_conditional_edges(
            "plot_approval_graph_state",
            self.is_plot_approval_complete,
            {
                "plot_approval_graph_state": "plot_approval_graph_state",
                "plot_generation_graph_state": "plot_generation_graph_state",
                "paper_writer_graph_state": "paper_writer_graph_state",
            },
        )
        builder.add_edge("plot_suggestion_graph_state", "plot_approval_graph_state")
        builder.add_edge("plot_generation_graph_state", "paper_writer_graph_state")

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
                "plot_approval_graph_state",
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
        return "plot_suggestion_graph_state"

    def is_plot_approval_complete(self, state: AgentState):
        """Check if plot approval workflow is complete."""
        suggested_plots = getattr(state, "suggested_plots", [])
        if not suggested_plots:
            return "paper_writer_graph_state"

        approved_plots = [p for p in suggested_plots if p.approved]
        if approved_plots:
            return "plot_generation_graph_state"

        all_decided = all("approved" in p for p in suggested_plots)
        if all_decided:
            return "paper_writer_graph_state"
        else:
            return "plot_approval_graph_state"

    def is_generate_review_complete(self, state: AgentState, config: dict) -> str:
        """
        Checks if review of generation phase is complete based on number of revisions.

        :param state: state of agent.
        :return: next state to go.
        """
        if config["configurable"]["instruction"]:
            return "writer_manual_review_graph_state"
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
        img = self.graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        display(Image(img))

        with open("kiroku_graph.png", "wb") as f:
            f.write(img)


class KirokuUI:
    def __init__(self, working_dir: Path):
        """
        Initialize KirokuUI with dual plot workflows:
        - File-based workflow (with approval)
        - Standalone UI workflow (instant)
        """
        self.working_dir = working_dir
        self.first = True
        self.next_state = -1
        self.references = []
        self.state_values = PaperConfig()
        self.writer = None

        # Initialize PlotSuggester independently
        # This allows standalone plot generation without YAML upload
        self.plotter = self._initialize_plotter()

        # Standalone plot gallery (for UI-based workflow)
        self.plot_gallery = []
        self.selected_plot_ids = set()

        # File-based plot components (will be initialized in create_ui)
        self.plot_approval_section = None
        self.plot_suggestions_display = None
        self.plot_checkboxes = []
        self.plot_checkboxes_section = None

        logging.info("✓ KirokuUI initialized")

    def _initialize_plotter(self):
        """
        Initialize PlotSuggester independently of DocumentWriter.
        This enables plot generation without uploading YAML.

        Returns:
            PlotSuggester instance or None if initialization fails
        """
        try:
            # Create standalone model for plotting
            model = ChatOpenAI(
                model=config("OPENAI_MODEL_NAME", default="gpt-5-nano"),
                temperature=1,
                openai_api_key=config("OPENAI_API_KEY"),
            )

            plotter = PlotSuggester(model)
            logging.info("✓ PlotSuggester initialized successfully")
            return plotter

        except Exception as e:
            logging.error(f"Failed to initialize PlotSuggester: {e}")
            logging.warning("Plot generation will be unavailable")
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

    def step(self, instruction: str, state_values: PaperConfig = None) -> str:
        """
        Performs one step of the graph invocation, stopping at the next break point.
        :param instruction: instruction to execute.
        :param state_values: initial state values or None if continuing.
        :return: draft of the paper.
        """
        config = {"instruction": instruction}
        return self.writer.invoke(state_values, config)
    def _get_logs_data(self):
        """Returns raw log data instead of UI components."""
        if not self.writer:
            return []

        state = self.writer.get_state()
        logs = state.values.get("workflow_logs", [])
        return logs
    # def _get_log_updates(self):
    #     if not self.writer:
    #         return [gr.update(visible=False)] * PREINITIALIZED_COMPONENTS, [
    #             gr.update()
    #         ] * PREINITIALIZED_COMPONENTS

    #     state = self.writer.get_state()
    #     logs = state.values.get("workflow_logs", [])
    #     accordion_updates = []
    #     markdown_updates = []

    #     # Group logs by step
    #     grouped_logs = {}
    #     steps_order = []
    #     for log in logs:
    #         if log.step not in grouped_logs:
    #             steps_order.append(log.step)
    #             grouped_logs[log.step] = []
    #         grouped_logs[log.step].append(log)

    #     for i in range(PREINITIALIZED_COMPONENTS):
    #         if i < len(steps_order):
    #             step_name = steps_order[i]
    #             step_logs = grouped_logs[step_name]

    #             content = ""
    #             for log in step_logs:
    #                 content += f"### {log.side}\n{log.message}\n\n---\n\n"

    #             accordion_updates.append(gr.update(visible=True, label=step_name))
    #             markdown_updates.append(gr.update(value=content))
    #         else:
    #             accordion_updates.append(gr.update(visible=False))
    #             markdown_updates.append(gr.update(value=""))

    #     return accordion_updates, markdown_updates

    # def update(self, instruction):
    #     """
    #     Updates state upon submitting an instruction or updating references.
    #     :param instruction: instruction to be executed.
    #     :return: new draft, atlas message and making input object non-interactive.
    #     """
    #     draft = self.step(instruction)
    #     state = self.writer.get_state()
    #     current_state = state.values["state"]
    #     try:
    #         next_state = state.next[0]
    #     except:
    #         next_state = "NONE"

    #     # if state is in reflection stage, draft to be shown is in the critique field.
    #     if (
    #         current_state == "reflection_reviewer_graph_state"
    #         and next_state == "additional_reflection_instructions_graph_state"
    #     ):
    #         draft = state.values["critique"]

    #     # if next state is going to generate citations, we populate the references
    #     # for the Tab References.
    #     if next_state == "generate_citations_graph_state":
    #         self.references = state.values.get("references", []).split("\n")

    #     # if we have reached the end, we will save everything.
    #     if next_state == END or next_state == "NONE":
    #         dir = os.path.splitext(self.filename)[0]
    #         logging.warning(f"saving final draft in {dir}")
    #         self.save_as()

    #     self.next_state = next_state

    #     # processed_draft = self._process_images_for_gradio(draft)

    #     plot_section_visible = next_state == "plot_approval_graph_state"
    #     plot_suggestions_content = "No plot suggestions available."
    #     checkbox_updates = [gr.update(visible=False)] * 5

    #     if plot_section_visible and hasattr(state, "values"):
    #         suggested_plots = state.values.get("suggested_plots", [])
    #         # if suggested_plots:
    #         #     plot_suggestions_content = self._format_plots_for_ui(suggested_plots)
    #         #     checkbox_updates = self._update_plot_checkboxes(suggested_plots)

    #     log_accordions, log_markdowns = self._get_log_updates()

    #     return (
    #         processed_draft,
    #         self.atlas_message(next_state),
    #         gr.update(interactive=False),
    #         gr.update(visible=plot_section_visible),
    #         gr.update(value=plot_suggestions_content),
    #         *tuple(checkbox_updates),
    #         *log_accordions,
    #         *log_markdowns,
    #     )
    def update(self, instruction):
        """Updates state and returns DATA for the renders."""
        draft = self.step(instruction)
        state = self.writer.get_state()
        current_state = state.values["state"]

        try:
            next_state = state.next[0]
        except:
            next_state = "NONE"

        if (current_state == "reflection_reviewer_graph_state" and
            next_state == "additional_reflection_instructions_graph_state"):
            draft = state.values["critique"]

        if next_state == "generate_citations_graph_state":
            self.references = state.values.get("references", []).split("\n")

        if next_state == END or next_state == "NONE":
            self.save_as()

        self.next_state = next_state

        # Get simplified data for the UI
        logs_data = self._get_logs_data()

        # Determine plot visibility
        plot_section_visible = next_state == "plot_approval_graph_state"

        # IMPORTANT: Return clean data structure for the state
        return (
            draft,                              # markdown output
            self.atlas_message(next_state),     # echo output
            gr.update(interactive=False),       # inp update
            logs_data,                          # -> logs_state
            gr.update(visible=plot_section_visible) # plot section visibility
        )
    # def atlas_message(self, state):
    #     """
    #     Returns the Echo message for a given state.
    #     :param state: Next state of the multi-agent system.
    #     :return:
    #     """
    #     message = {
    #         "suggest_title_review_graph_state": "Please suggest review instructions for the title.",
    #         "topic_sentence_manual_review_graph_state": "Please suggest review instructions for the topic sentences.",
    #         "writer_manual_reviewer_graph_state": "Please suggest review instructions for the main draft.",
    #         "additional_reflection_instructions_graph_state": "Please provide additional instructions for the overall paper review.",
    #         "generate_citations_graph_state": "Please look at the references tab and confirm the references.",
    #         "plot_approval_graph_state": "Please review the suggested plots and approve or reject them. Check the suggested plots below.",
    #     }

    #     instruction = message.get(state, "")
    #     if instruction or state == "generate_citations_graph_state":
    #         if state == "generate_citations_graph_state":
    #             return instruction
    #         return instruction + " Type <RETURN> when done."
    #     return "We have reached the end."
    def atlas_message(self, state):
        message = {
            "suggest_title_review_graph_state": "Please suggest review instructions for the title.",
            "topic_sentence_manual_review_graph_state": "Please suggest review instructions for the topic sentences.",
            "writer_manual_reviewer_graph_state": "Please suggest review instructions for the main draft.",
            "additional_reflection_instructions_graph_state": "Please provide additional instructions for the overall paper review.",
            "generate_citations_graph_state": "Please look at the references tab and confirm the references.",
            "plot_approval_graph_state": "Please review the suggested plots.",
        }
        instruction = message.get(state, "")
        if instruction or state == "generate_citations_graph_state":
            return instruction + (" Type <RETURN> when done." if state != "generate_citations_graph_state" else "")
        return "We have reached the end."
    # def initial_step(self) -> tuple:
    #     """
    #     Performs initial step, in which we need to providate a state to the graph.
    #     :return: draft and Echo message.
    #     """
    #     state_values = self.state_values.model_copy(deep=True)
    #     if self.state_values.suggest_title:
    #         state_values.state = "suggest_title_graph_state"
    #     else:
    #         state_values.state = "topic_sentence_writer_graph_state"
    #     draft = self.step("", state_values)
    #     state = self.writer.get_state()
    #     current_state = state.values["state"]
    #     try:
    #         next_state = state.next[0]
    #     except:
    #         next_state = "NONE"
    #     # processed_draft = self._process_images_for_gradio(draft)

    #     log_accordions, log_markdowns = self._get_log_updates()

    #     # Defaults for plot stuff
    #     show_plot_approval = False
    #     plot_suggestions_md = ""
    #     checkbox_updates = [gr.update(visible=False, value=False) for _ in range(5)]

    #     return (
    #         processed_draft,
    #         self.atlas_message(next_state),
    #         gr.update(
    #             value="",
    #             interactive=next_state not in [END, "generate_citations", "NONE"],
    #         ),
    #         gr.update(visible=show_plot_approval),
    #         gr.update(value=plot_suggestions_md),
    #         *checkbox_updates,
    #         *log_accordions,
    #         *log_markdowns,
    #     )
    def initial_step(self) -> tuple:
        """Performs initial step and returns data for states."""
        state_values = self.state_values.model_copy(deep=True)
        if self.state_values.suggest_title:
            state_values.state = "suggest_title_graph_state"
        else:
            state_values.state = "topic_sentence_writer_graph_state"

        draft = self.step("", state_values)
        state = self.writer.get_state()

        try:
            next_state = state.next[0]
        except:
            next_state = "NONE"

        logs_data = self._get_logs_data()

        return (
            draft,
            self.atlas_message(next_state),
            gr.update(value="", interactive=next_state not in [END, "generate_citations", "NONE"]),
            logs_data, # -> logs_state
            gr.update(visible=False) # Plot section
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
        # draft = self._process_images_for_gradio(draft)
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
        pass
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
        return self.update("")

    # def _format_plots_for_ui(self, plots: list) -> str:
    #     """Format plots for display in the plot approval UI."""
    #     if not plots:
    #         return "No plots suggested."

    #     content = "## Suggested Plots\n\n"
    #     for i, plot in enumerate(plots, 1):
    #         status = "✅ APPROVED" if plot.approved else "❌ Pending approval"
    #         content += f"### Plot {i}: {plot.description} [{status}]\n\n"
    #         content += f"**Purpose:** {plot.rationale}\n\n"

    #         full_code = plot.code
    #         content += f"**Full Code:**\n```python\n{full_code}\n```\n\n"

    #     content += "**Instructions:**\n"
    #     content += (
    #         "- Use individual checkboxes above to select which plots to approve\n"
    #     )
    #     content += (
    #         "- Click 'Apply Individual Selections' to apply your checkbox choices\n"
    #     )
    #     content += (
    #         "- Or use the 'Approve All' / 'Reject All' buttons for bulk actions\n"
    #     )
    #     content += "- Old text commands still work: 'approve 1,3,5' or 'reject 2,4'\n"
    #     return content

    def _handle_plot_decision(self, decision: str) -> gr.Textbox:
        """Helper method to handle plot approval/rejection decisions."""
        return gr.Textbox(value=decision, interactive=True)

    # def _update_plot_checkboxes(self, suggested_plots: list):
    #     """Update the visibility and labels of plot checkboxes."""
    #     updates = []
    #     for i, checkbox in enumerate(self.plot_checkboxes):
    #         if i < len(suggested_plots):
    #             plot = suggested_plots[i]
    #             updates.append(
    #                 gr.update(
    #                     visible=True,
    #                     label=f"✓ Plot {i + 1}: {plot.description}",
    #                     value=plot.approved,
    #                 )
    #             )
    #         else:
    #             updates.append(gr.update(visible=False))
    #     return updates

    # def _apply_checkbox_selections(self, *checkbox_values):
    #     """Apply individual checkbox selections to plot approval."""
    #     if hasattr(self, "writer") and self.writer is not None:
    #         state = self.writer.get_state()
    #         suggested_plots = state.values.get("suggested_plots", [])

    #         logging.info(f"Applying checkbox selections: {checkbox_values}")
    #         logging.info(f"Found {len(suggested_plots)} suggested plots")

    #         # Update approval status based on checkbox values
    #         for i, checkbox_value in enumerate(checkbox_values):
    #             if i < len(suggested_plots):
    #                 old_status = suggested_plots[i].approved
    #                 suggested_plots[i].approved = bool(checkbox_value)
    #                 logging.info(
    #                     f"Plot {i + 1}: {old_status} -> {bool(checkbox_value)}"
    #                 )

    #         state.values["suggested_plots"] = suggested_plots
    #         self.writer.update_state(state)

    #         # Continue to next step
    #         return self.update("apply individual selections")
    #     return self.update("")

    # def _process_images_for_gradio(self, draft: str) -> str:
    #     """Process image paths in markdown to make them work better in Gradio."""
    #     import re
    #     import base64

    #     logging.info(f"Processing draft for Gradio image display, length: {len(draft)}")

    #     # Convert relative paths to data URIs for Gradio compatibility
    #     def replace_image_path(match):
    #         alt_text = match.group(1)
    #         path = match.group(2).strip()

    #         logging.info(f"Processing image: [{alt_text}]({path})")

    #         # If it's already a data URI or absolute HTTP path, leave it
    #         if path.startswith(("data:", "http:", "https:")):
    #             logging.info(f"Skipping already processed path: {path[:50]}...")
    #             return match.group(0)

    #         # Handle different path formats
    #         absolute_path = None

    #         # Normalize path separators first
    #         normalized_path = path.replace("\\", "/")

    #         # Handle different path formats
    #         if normalized_path.startswith("../images/"):
    #             filename = normalized_path.replace("../images/", "")
    #             absolute_path = os.path.join(self.working_dir, "images", filename)
    #         elif normalized_path.startswith("proj/images/"):
    #             filename = normalized_path.replace("proj/images/", "")
    #             absolute_path = os.path.join(self.working_dir, "images", filename)
    #             logging.info(f"Proj path: {filename} -> {absolute_path}")
    #         elif normalized_path.startswith("images/"):
    #             filename = normalized_path.replace("images/", "")
    #             absolute_path = os.path.join(self.working_dir, "images", filename)
    #             logging.info(f"Images path: {filename} -> {absolute_path}")
    #         elif normalized_path.startswith("/images/"):
    #             # Handle absolute path starting with /images/
    #             filename = normalized_path.replace("/images/", "")
    #             absolute_path = os.path.join(self.working_dir, "images", filename)
    #             logging.info(f"Absolute images path: {filename} -> {absolute_path}")
    #         elif os.path.isabs(path):
    #             absolute_path = path
    #             logging.info(f"Absolute path: {absolute_path}")
    #         else:
    #             absolute_path = os.path.join(self.working_dir, path)
    #             logging.info(f"Generic relative path: {path} -> {absolute_path}")

    #             # If that doesn't work, try in images subdirectory
    #             if not os.path.exists(absolute_path):
    #                 absolute_path = os.path.join(self.working_dir, "images", path)
    #                 logging.info(f"Trying in images subdir: {path} -> {absolute_path}")

    #             # Final fallback: try just the filename in images directory
    #             if not os.path.exists(absolute_path):
    #                 filename_only = os.path.basename(path)
    #                 absolute_path = os.path.join(
    #                     self.working_dir, "images", filename_only
    #                 )
    #                 logging.info(
    #                     f"Trying filename only: {filename_only} -> {absolute_path}"
    #                 )

    #         if absolute_path:
    #             logging.info(f"Absolute path: {absolute_path}")

    #             # Try to convert to data URI
    #             try:
    #                 if os.path.exists(absolute_path):
    #                     with open(absolute_path, "rb") as img_file:
    #                         img_data = img_file.read()
    #                         # Determine MIME type from extension
    #                         ext = os.path.splitext(absolute_path)[1].lower()
    #                         mime_types = {
    #                             ".png": "image/png",
    #                             ".jpg": "image/jpeg",
    #                             ".jpeg": "image/jpeg",
    #                             ".gif": "image/gif",
    #                             ".svg": "image/svg+xml",
    #                         }
    #                         mime_type = mime_types.get(ext, "image/png")

    #                         # Encode as base64 data URI
    #                         b64_data = base64.b64encode(img_data).decode("utf-8")
    #                         data_uri = f"data:{mime_type};base64,{b64_data}"
    #                         logging.info(
    #                             f"Successfully converted to data URI: {alt_text}"
    #                         )
    #                         return f"![{alt_text}]({data_uri})"
    #                 else:
    #                     logging.warning(f"Image file not found: {absolute_path}")

    #             except Exception as e:
    #                 logging.warning(f"Failed to process image {absolute_path}: {e}")

    #             # Fallback: use absolute path with forward slashes
    #             if os.path.exists(absolute_path):
    #                 absolute_path = absolute_path.replace(os.sep, "/")
    #                 logging.info(f"Using absolute path fallback: {absolute_path}")
    #                 return f"![{alt_text}]({absolute_path})"

    #         logging.warning(f"Could not process image path: {path}")
    #         return match.group(0)

    #     # Find and replace image references in markdown
    #     image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

    #     # Log all image matches before processing
    #     matches = re.findall(image_pattern, draft)
    #     if matches:
    #         logging.info(f"Found {len(matches)} image references in draft:")
    #         for i, (alt, path) in enumerate(matches, 1):
    #             logging.info(f"  {i}: [{alt}]({path})")
    #     else:
    #         logging.info("No image references found in draft")

    #     processed_draft = re.sub(image_pattern, replace_image_path, draft)

    #     # Check if any processing actually occurred
    #     if processed_draft != draft:
    #         logging.info("Draft was modified during image processing")
    #         # Log first data URI if any
    #         if "data:image/" in processed_draft:
    #             logging.info("Successfully converted at least one image to data URI")
    #         else:
    #             logging.warning("No data URIs found in processed draft")
    #     else:
    #         logging.info("Draft was not modified during image processing")

    #     logging.info(f"Finished processing images for Gradio")
    #     return processed_draft

    # ========================================
    # HELPER METHODS
    # ========================================

    def _handle_plot_decision(self, decision: str):
        """
        Handle bulk approval/rejection of plots.

        Args:
            decision: "approve all" or "reject all"

        Returns:
            Gradio update for input field
        """
        return gr.update(value=decision)


    def _apply_checkbox_selections(self, *checkbox_states):
        """
        Apply individual checkbox selections to plot approval.

        Args:
            *checkbox_states: Boolean values for each checkbox

        Returns:
            Updates for all UI components
        """
        # Get current state
        state = self.writer.get_state()
        suggested_plots = state.values.get("suggested_plots", [])

        # Update approval status based on checkboxes
        for i, is_checked in enumerate(checkbox_states):
            if i < len(suggested_plots):
                suggested_plots[i].approved = is_checked

        # Update state
        state.values["suggested_plots"] = suggested_plots
        self.writer.update_state(state)

        # Continue with plot generation
        instruction = "apply individual selections"
        return self.update(instruction)

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

    def generate_multiple_plots(self, plot_prompt: str, num_plots: int):
        """
        Generate multiple plot variations for standalone plot tab.

        Args:
            plot_prompt: User's description of desired plot
            num_plots: Number of variations to generate

        Returns:
            Updates for plot components, code display, and status
        """
        try:
            # Get paper draft if available (optional)
            draft = ""
            if self.writer is not None:
                state = self.writer.get_state()
                draft = state.values.get("draft", "")

            has_prompt = bool(plot_prompt.strip())
            has_draft = bool(draft.strip())

            # Validate: need at least one input
            if not has_prompt and not has_draft:
                return self._create_error_message(
                    "⚠️ No input provided\n\n"
                    "Please either:\n"
                    "• Enter a plot description, or\n"
                    "• Generate paper content first\n\n"
                    "Both together gives best results!"
                )

            # Determine mode
            if has_draft and has_prompt:
                mode = "📄+📝 Paper-contextualized plot"
            elif has_draft:
                mode = "📄 Paper-based plot (no description)"
            else:
                mode = "📝 Description-only plot"

            logging.info(f"Generating {num_plots} plot(s) | Mode: {mode}")

            # Generate plots
            plot_results = []
            for i in range(num_plots):
                logging.info(f"Generating variation {i+1}/{num_plots}...")

                try:
                    fig, code = self.plotter.suggest_plot(
                        paper_content=draft if has_draft else "",
                        user_prompt=plot_prompt if has_prompt else "",
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
                    plot_results.append(plot_version)

                    logging.info(f"✓ Variation {i+1} generated")

                except Exception as e:
                    logging.error(f"Failed to generate variation {i+1}: {e}")
                    # Continue with other plots
                    continue

            success_count = len(plot_results)
            logging.info(f"✓ Complete: {success_count}/{num_plots} successful")

            return self.render_gallery(num_plots)

        except Exception as e:
            logging.error("Critical error in generate_multiple_plots", exc_info=True)
            # return self._create_error_message(f"Critical Error:\n{str(e)}")
        finally:
            if self.plotter:
                self.plotter.cleanup()

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


    def create_ui(self) -> None:
        with gr.Blocks(
            theme=gr.themes.Default(), fill_height=True
        ) as self.kiroku_agent:
            with gr.Tab("Initial Instructions"):
                with gr.Row():
                    file = gr.File(file_types=[".yaml"], scale=1)
                    js = gr.JSON(scale=5)
                _, _, files_preview = self._make_files_preview()
            with gr.Tab("Document Writing"):
                out = gr.Textbox(label="Echo")
                inp = gr.Textbox(placeholder="Instruction", label="Rider")
                markdown = gr.Markdown("")

                with gr.Group(visible=False) as self.plot_approval_section:
                    gr.Markdown("### Plot Suggestions")
                    self.plot_suggestions_display = gr.Markdown(
                        "No plot suggestions available."
                    )

                    with gr.Column() as self.plot_checkboxes_section:
                        self.plot_checkboxes = []
                        for i in range(5):
                            checkbox = gr.Checkbox(
                                label=f"Plot {i + 1}",
                                value=False,
                                visible=False,
                                interactive=True,
                            )
                            self.plot_checkboxes.append(checkbox)

                    with gr.Row():
                        approve_plots_btn = gr.Button(
                            "✓ Approve All Plots", variant="primary"
                        )
                        reject_plots_btn = gr.Button(
                            "✗ Reject All Plots", variant="secondary"
                        )
                        apply_checkboxes_btn = gr.Button(
                            "Apply Individual Selections", variant="secondary"
                        )

                doc = gr.Button("Save")
            with gr.Tab("References") as self.ref_block:
                ref_list = [
                    gr.Checkbox(
                        value=False, visible=False, label=False, interactive=True
                    )
                    for _ in range(1000)
                ]
                submit_ref_list = gr.Button("Submit", visible=False)

            # ========================================
            # TAB 4: Standalone Plot Generation (NEW!)
            # ========================================
            with gr.Tab("Plot Generation") as self.plot_tab:
                gr.Markdown("""
                # 🎨 Instant Plot Generator

                Generate publication-quality plots instantly - **no document required!**

                **Two modes:**
                1. **With paper context**: If you've generated a document, plots will be tailored to it
                2. **Standalone**: Just describe what you want to visualize
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Plot Description")

                        plot_prompt = gr.Textbox(
                            label="What do you want to visualize?",
                            placeholder="e.g., Show the distribution of response times across experimental conditions with error bars",
                            lines=4,
                            info="Describe your desired plot in natural language"
                        )

                        gr.Markdown("""
                        **Examples:**
                        - "Compare sales trends across regions over the last 5 years"
                        - "Show correlation heatmap between all variables"
                        - "Create violin plots of reaction times by age group"
                        """)

                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Settings")

                        num_plots_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Number of variations",
                            info="Generate multiple design options"
                        )

                        generate_btn = gr.Button(
                            "✨ Generate Plots",
                            variant="primary",
                            size="lg",
                            scale=2
                        )

                        gr.Markdown("""
                        **Status:**
                        - 📄 Paper available: Contextualized plots
                        - 📝 Description only: Generic plots
                        """)

                # Status banner
                with gr.Row():
                    status_text = gr.Textbox(
                        label="📍 Generation Status",
                        value="Ready! Enter a description and click Generate",
                        interactive=False,
                        max_lines=2
                    )

                gr.Markdown("---")
                gr.Markdown("### 🖼️ Generated Plots")

                # Plot gallery
                plot_components = []
                plot_imgs = []
                plot_checkboxes = []

                with gr.Row():
                    for i in range(NUM_PLOTS):
                        with gr.Column():
                            with gr.Group():
                                plot_img = gr.Plot(
                                    label=f"Variation {i+1}",
                                    visible=False
                                )
                                plot_checkbox = gr.Checkbox(
                                    label=f"✓ Select",
                                    value=False,
                                    visible=False,
                                    interactive=True,
                                    scale=0
                                )

                                plot_imgs.append(plot_img)
                                plot_checkboxes.append(plot_checkbox)
                                plot_components.extend([plot_img, plot_checkbox])

                gr.Markdown("---")
                gr.Markdown("### 💾 Export Selected Plots")

                with gr.Row():
                    with gr.Column():
                        selected_code = gr.Code(
                            label="Python Code for Selected Plots",
                            language="python",
                            lines=20,
                            value="""# Select plots above to view their code
# How to use:
# 1. Check the boxes under plots you like
# 2. Copy this code to your script
# 3. Customize as needed!
"""
                        )

                        gr.Markdown("""
**💡 Tips:**
- Selected code is ready to run
- Modify colors, labels, styles as needed
- Save figures with `fig.savefig('plot.png', dpi=300)`
                        """)

            # ========================================
            # TAB 5: Instruction Log
            # ========================================
            with gr.Tab("Instruction Log"):
                gr.Markdown("""
                # 📋 Workflow Logs
                View detailed logs of all AI interactions and decisions.
                """)

                _, self.log_accordions, self.log_markdowns = (
                    self._logs_viewer_elements()
                )

            # ========================================
            # EVENT HANDLERS - Document Writing Tab
            # ========================================

            # Submit instruction in document writing
            # inp.submit(
            #     self.update,
            #     inputs=[inp],
            #     outputs=[
            #         markdown,
            #         out,
            #         inp,
            #         self.plot_approval_section,
            #         self.plot_suggestions_display,
            #         *self.plot_checkboxes,
            #         *self.log_accordions,
            #         *self.log_markdowns,
            #     ],
            # ).then(
            #     lambda: gr.update(
            #         value="",
            #         interactive=self.next_state not in [END, "generate_citations", "NONE"],
            #     ),
            #     outputs=[inp],
            # ).then(
            #     self.update_refs,
            #     outputs=[submit_ref_list, *ref_list]
            # )

            # Upload YAML file
            # file.upload(
            #     self.process_file,
            #     inputs=[file],
            #     outputs=[*files_preview, js, inp]
            # ).then(
            #     self.initial_step,
            #     outputs=[
            #         markdown,
            #         out,
            #         inp,
            #         self.plot_approval_section,
            #         self.plot_suggestions_display,
            #         *self.plot_checkboxes,
            #         *self.log_accordions,
            #         *self.log_markdowns,
            #     ],
            # ).then(
            #     lambda: gr.update(placeholder="", interactive=True),
            #     outputs=[inp]
            # )

            # Save document
            # doc.click(self.save_as, outputs=[out])

            # Submit references
            # submit_ref_list.click(
            #     self.submit_ref_list,
            #     inputs=ref_list,
            #     outputs=[
            #         markdown,
            #         out,
            #         inp,
            #         self.plot_approval_section,
            #         self.plot_suggestions_display,
            #         *self.plot_checkboxes,
            #         *self.log_accordions,
            #         *self.log_markdowns,
            #     ],
            # )

            # Plot approval buttons (file-based workflow)
            # approve_plots_btn.click(
            #     lambda: self._handle_plot_decision("approve all"),
            #     outputs=[inp]
            # )

            # reject_plots_btn.click(
            #     lambda: self._handle_plot_decision("reject all"),
            #     outputs=[inp]
            # )

            # apply_checkboxes_btn.click(
            #     self._apply_checkbox_selections,
            #     inputs=self.plot_checkboxes,
            #     outputs=[
            #         markdown,
            #         out,
            #         inp,
            #         self.plot_approval_section,
            #         self.plot_suggestions_display,
            #         *self.plot_checkboxes,
            #         *self.log_accordions,
            #         *self.log_markdowns,
            #     ],
            # )

            # ========================================
            # EVENT HANDLERS - Plot Generation Tab (Standalone)
            # ========================================

            # Generate plots button
            generate_btn.click(
                fn=self.generate_multiple_plots,
                inputs=[plot_prompt, num_plots_slider],
                outputs=[*plot_components, selected_code, status_text]
            )

            # Checkbox changes for plot selection
            for idx, checkbox in enumerate(plot_checkboxes):
                checkbox.change(
                    fn=lambda is_checked, slot_idx=idx: self.handle_checkbox_change(
                        slot_idx, is_checked
                    ),
                    inputs=[checkbox],
                    outputs=[*plot_checkboxes, selected_code, status_text]
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
        self.kiroku_agent.launch()

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


