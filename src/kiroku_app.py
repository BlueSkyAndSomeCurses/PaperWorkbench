# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import logging
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import gradio as gr
import markdown
import polars as pl
import yaml
from gradio.components.html import HTML
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.states import *
from src.utils.models import PaperConfig

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from decouple import config

PREINITIALIZED_COMPONENTS = 10

logging.basicConfig(level=logging.WARNING)


class DocumentWriter:
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
            model=model_name,
            temperature=temperature,
            openai_api_key=config("OPENAI_API_KEY"),
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
            builder.add_edge("analyze_relevant_files", "suggest_title")
            builder.add_edge("suggest_title", "suggest_title_review")
        else:
            builder.add_edge("analyze_relevant_files", "internet_search")
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

        builder.set_entry_point("analyze_relevant_files")

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
        if state.revision_number <= state.max_revisions:
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


class KirokuUI:
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.first = True
        self.next_state = -1
        self.references = []
        self.state_values = PaperConfig()

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
        state_values = self.state_values.model_copy(deep=True)
        if self.state_values.suggest_title:
            state_values.state = "suggest_title"
        else:
            state_values.state = "topic_sentence_writer"
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

        logging.info(f"State values: \n\n{self.state_values}")

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
                doc = gr.Button("Save")
            with gr.Tab("References") as self.ref_block:
                ref_list = [
                    gr.Checkbox(
                        value=False, visible=False, label=False, interactive=True
                    )
                    for _ in range(1000)
                ]
                submit_ref_list = gr.Button("Submit", visible=False)

            inp.submit(self.update, inp, [markdown, out, inp]).then(
                lambda: gr.update(
                    value="",
                    interactive=self.next_state
                    not in [END, "generate_citations", "NONE"],
                ),
                [],
                inp,
            ).then(self.update_refs, [], [submit_ref_list, *ref_list])
            file.upload(self.process_file, file, [*files_preview, js, inp]).then(
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
        )  # allowed_paths=[working_dir])

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
                    in [".yaml", ".yml", ".md", ".tex", ".html", ".csv", ".parquet"]
                    and file.name == file_name
                ):
                    relevant_files.append(
                        RelevantFile(file_path=file, description=description)
                    )

        return relevant_files

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
