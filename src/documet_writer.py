from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from src.utils.models import PaperConfig
from src.agents.suggest_plot import PlotSuggestionAgent
from src.agents.generate_plot import PlotGenerationAgent

from src.agents.states import *
from src.utils.models import PaperConfig

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from decouple import config

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
                PlotSuggestionAgent(self.model_m),
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