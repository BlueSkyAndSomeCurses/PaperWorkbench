# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import logging
import os
import re
import shutil
import pandas as pd
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


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


class KirokuUI:
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.first = True
        self.next_state = -1
        self.references = []
        self.state_values = PaperConfig()
        self.writer = None
        self.plotter = None

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

    def handle_plot_suggestion(self, user_data: pd.DataFrame):
        if self.writer is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.text(0.5, 0.5, 'Please upload a YAML configuration file first\nin the "Initial Instructions" tab.', 
                   ha='center', va='center', fontsize=12, wrap=True)
            ax.axis('off')
            code = "# No configuration loaded - please upload a YAML file first"
            return fig, code

        try:
            state = self.writer.get_state()
            draft = state.values.get("draft", "")
            
            if not draft:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.text(0.5, 0.5, 'No draft available yet.\nPlease generate paper content first.', 
                    ha='center', va='center', fontsize=12)
                ax.axis('off')
                code = "# No draft available"
                return fig, code
            
            sample_data = pd.DataFrame({
                'x': [1, 2, 3, 4, 5],
                'y': [2, 4, 3, 5, 4]
            })
            
            if user_data is not None:
                try:
                    logging.info("Generating plot suggestion on user's input...")
                    data = pd.DataFrame(user_data)
                except:
                    logging.info("Generating plot suggestion on sampled data...")
                    data = sample_data
            
            fig, code = self.plotter.suggest_plot(draft, data)            
            logging.info("Plot generated successfully")
            
            print('Returned type:', type(fig))  # <class 'matplotlib.figure.Figure'>
            # assert isinstance(fig, matplotlib.figure.Figure), f'Returned {type(fig)} instead of Figure!'
            return gr.update(value=fig), code
            
        except Exception as e:
            logging.error(f"Error in plot suggestion: {e}", exc_info=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            error_msg = str(e)
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            ax.text(0.5, 0.5, f'Error generating plot:\n{error_msg}', 
                ha='center', va='center', fontsize=10, color='red', wrap=True)
            ax.axis('off')
            return fig, f"# Error: {str(e)}"

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

            with gr.Tab("Plot Suggestion") as self.plottab:
                with gr.Row():
                    with gr.Column():
                        data_input = gr.Dataframe(label="Example Data (paste or upload first 5 rows, set columns)", datatype="str")
                        plotbutton = gr.Button("Suggest Plot", variant="primary", size="lg")
                        gr.Markdown("""
                        Click the button above to generate a plot suggestion based on your paper content.
                        The system will analyze your draft and create a relevant visualizations.
                        """)
                    
                with gr.Row():
                    with gr.Column(scale=1):
                        plotimage = gr.Plot(label="Suggested Plot")
                    with gr.Column(scale=1):
                        plotcode = gr.Code(label="Python Code", language="python", lines=20)
                
                plotbutton.click(
                    self.handle_plot_suggestion,
                    inputs=[data_input],
                    outputs=[plotimage, plotcode]
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
        )  # allowed_paths=[working_dir])


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
