# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import json
import logging
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.utils.file_handlers import handle_file_reading_for_model
from src.utils.models import AgentState, RelevantaFileApplication, RelevantFile

from .gen_citations import insert_references
from .prompts import (
    ABSTRACT_WRITER_PROMPT,
    INTERNET_SEARCH_PROMPT,
    LATEX_CONVERSION_PROMPT,
    PAPER_WRITER_PROMPT,
    REFERENCES_PROMPT,
    REFLECTION_REVIEWER_PROMPT,
    RESEARCH_CRITIQUE_PROMPT,
    SYSTEM_SUMARIZE_PROMPT,
    TASK_TEMPLATE,
    TITLE_PROMPT,
    TOPIC_SENTENCE_PROMPT,
    TOPIC_SENTENCE_REVIEW_PROMPT,
    USER_SUMARIZE_PROMPT,
    WRITER_REVIEW_PROMPT,
)
from .search import *


class State:
    def __init__(self, model: ChatOpenAI, name: str) -> None:
        self.model = model
        self.name = name


class AnalyzeRelevantFiles(State):
    def __init__(state, model: ChatOpenAI) -> None:
        super().__init__(model, "analyze_relevant_files")

    def run(self, state: AgentState, config: dict) -> dict:
        logging.info(f"state {self.name}, state config: {state}")

        def process_file(relevant_file: RelevantFile) -> RelevantFile:
            logging.info(relevant_file.file_path)
            messages = [
                SystemMessage(content=SYSTEM_SUMARIZE_PROMPT),
                HumanMessage(
                    content=USER_SUMARIZE_PROMPT.format(
                        title=state.title,
                        instructions=state.instructions,
                        hypothesis=state.hypothesis,
                        desc=relevant_file.description,
                        document=handle_file_reading_for_model(relevant_file.file_path),
                    )
                    + "\n".join(
                        [
                            f"{section_name}: {desc}"
                            for section_name, desc in state.section_names.items()
                        ]
                    )
                ),
            ]

            response = self.model.invoke(
                messages, response_format={"type": "json_object"}
            )

            json_response = json.loads(response.content)
            for section_name, application in json_response.items():
                relevant_file.application.append(
                    RelevantaFileApplication(
                        stage_name=section_name, application_desc=application
                    )
                )
            return relevant_file

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_file, relevant_file)
                for relevant_file in state.relevant_files
            ]

            return {
                "relevant_files": [future.result() for future in as_completed(futures)]
            }


class SuggestTitle(State):
    def __init__(self, model):
        super().__init__(model, "suggest_title")

    def run(self, state: AgentState, config: dict) -> dict:
        """Node of graph that suggests a title for the paper.

        :param state: state of the agent.
        :return: fields 'title', 'draft' and 'messages' updated for the paper.
        """

        logging.info(f"state {self.name}: running")

        messages = state.messages
        if not messages:
            title = state.title
            area_of_paper = state.area_of_paper
            hypothesis = state.hypothesis

            messages = [
                SystemMessage(
                    content=TITLE_PROMPT.format(
                        area_of_paper=area_of_paper, title=title, hypothesis=hypothesis
                    )
                ),
                HumanMessage(
                    content=(
                        "Write the original title first. Then,"
                        "generate 10 thought provoking titles that "
                        "instigates reader's curiosity based on the given information"
                    )
                ),
            ]
        response = self.model.invoke(messages)
        messages.append(response)
        return {
            "state": self.name,
            "title": response.content,
            "messages": messages,
            "draft": response.content,
        }


class SuggestTitleReview(State):
    def __init__(self, model):
        super().__init__(model, "suggest_title_review")

    def run(self, state: AgentState, config: dict) -> dict:
        """Node of graph that suggests a title for the paper.

        :param state: state of the agent.
        :return: fields 'title', 'draft' and 'messages' updated for the paper.
        """

        logging.info(f"state {self.name}: running")

        messages = state.messages
        instruction = config["configurable"]["instruction"]
        if not instruction:
            human_message = HumanMessage(
                content="Just return the final title without any additional information"
            )
        else:
            human_message = HumanMessage(content=instruction)
        messages.append(human_message)
        response = self.model.invoke(messages)
        messages.append(response)
        if not instruction:
            messages = []
        title = response.content
        return {
            "state": self.name,
            "title": title,
            "messages": messages,
            "draft": response.content,
        }


class InternetSearch(State):
    def __init__(self, model):
        super().__init__(model, "internet_search")

    def create_task(
        self,
        title,
        hypothesis,
        area_of_paper,
        type_of_document,
        section_names,
        number_of_paragraphs,
        results,
        references,
    ):
        """
        Creates a writing task that will be executed by the agent.

        :param title: Title of the paper.
        :param hypothesis: Hypothesis of the paper, like "want to check if sky is blue".
        :param area_of_paper: Describes general field of knowledge of the paper.
        :param type_of_document: If document is a research paper, technical briefing, etc.
        :param section_names: List of sections for paper.
        :param number_of_paragraphs: List or Dict of number of paragraphs for each section.
        :param results: Results to be presented, if any.
        :param references: List of references to be used, if any.
        :return: prompt containing task to be executed.
        """
        if not hypothesis:
            hypothesis = "No paper hypothesis given."
        if not results:
            results = "No paper results given."
        if not references:
            references = (
                "No paper references given. "
                "Use 'research_plan' phase to get the references."
            )

        # number of sections should be equal to number of paragraphs per section.
        assert len(section_names) == len(number_of_paragraphs)
        if isinstance(number_of_paragraphs, dict):
            number_of_paragraphs = [
                number_of_paragraphs[section] for section in section_names
            ]
        sections = (
            ", ".join([f"'{section}'" for section in list(section_names)[:-1]])
            + f" and '{list(section_names)[-1]}'"
        )
        instruction = " ".join(
            [
                f"Section '{section}' will have {no_of_sentences} paragraphs."
                if no_of_sentences > 0
                else (
                    f"Section '{section}' will have no paragraphs, "
                    "as it will be filled later."
                )
                for (section, no_of_sentences) in zip(
                    section_names, number_of_paragraphs, strict=False
                )
            ]
        )
        task = TASK_TEMPLATE.format(
            title=title,
            type_of_document=type_of_document,
            area_of_paper=area_of_paper,
            sections=sections,
            instruction=instruction,
            hypothesis=hypothesis,
            results=results,
            references="\n".join(references),
        )
        return task

    def run(self, state: AgentState):
        """
        Performs a search on the internet on the topic of the paper.

        :param state: current state of the agent.
        :return: field 'content' added to state.
        """

        logging.info(f"state {self.name}: running")
        logging.info(f"state {self.name}: state {state}")

        queries = {"queries": []}
        task = self.create_task(
            title=state.title,
            hypothesis=state.hypothesis,
            area_of_paper=state.area_of_paper,
            type_of_document=state.type_of_document,
            section_names=state.section_names,
            number_of_paragraphs=state.number_of_paragraphs,
            results=state.results,
            references=state.references,
        )
        for _ in range(3):  # three attempts
            result = self.model.invoke(
                [
                    SystemMessage(
                        content=(
                            INTERNET_SEARCH_PROMPT.format(
                                number_of_queries=state.number_of_queries
                            )
                            + " You must only output the response in a plain list of queries "
                            "in the JSON format '{ \"queries\": list[str] }' and no other text. "
                            "You MUST only cite references that are in the references "
                            "section. "
                        )
                    ),
                    HumanMessage(content=task),
                ],
                response_format={"type": "json_object"},
            ).content
            # we need to add this because sometimes the LLM decides to put a header
            # in the json file.
            if result[:7] == "```json":
                result = result.split("\n")
                result = "\n".join(result[1:-1])
            content = state.content
            try:
                queries = json.loads(result)
                break
            except:
                logging.warning(f"state {self.name}: could not extract query {result}.")
        # finally, add to the queries all references that have http
        for ref in state.references:
            search_match = re.search(r"http.*(\s|$)", ref)
            if search_match:
                l, r = search_match.span()
                http_ref = ref[l:r]
                queries["queries"] = [http_ref, *queries["queries"]]
        if queries["queries"]:
            search, cache = search_query_ideas(query_ideas=queries, cache=set())
            content = content + search
        else:
            cache = set()
        return {
            "state": self.name,
            "content": content,
            "cache": cache,
            "task": task,
            "messages": [],
            "references": [],
        }


class TopicSentenceWriter(State):
    def __init__(self, model):
        super().__init__(model, "topic_sentence_writer")

    def run(self, state: AgentState):
        """
        Creates a bullet list plan for the paper with topic sentences.

        :param state: current state of the agent.
        :return: field 'plan' added to the state.
        """

        logging.info(f"state {self.name}: running")
        logging.info(f"{state}")
        task = state.task
        content = "\n\n".join(state.content)
        messages = state.messages
        if not messages:
            messages = [SystemMessage(content=TOPIC_SENTENCE_PROMPT)]
        messages.append(
            HumanMessage(
                content=f"""This is the content of a search on the internet for the paper:\n\n
                    {content}\n\n
                    {task}"""
            )
        )

        messages = [*messages, *self._generate_document_messages(state)]

        logging.info(messages)

        response = self.model.invoke(messages)
        plan = response.content.strip()
        if "```markdown" in plan:
            plan = "\n".join(plan.split("\n")[1:-1])
        # sometimes, the LLM just decide it will not obey the instructions to not
        # add references. So, we will remove them here.
        plan = plan.strip()
        search = re.search(r"## References", plan)
        if search:
            _, r = search.span()
            plan = plan[:r]
        messages.append(AIMessage(content=plan))
        return {"state": self.name, "plan": plan, "draft": plan, "messages": messages}

    def _generate_document_messages(self, state: AgentState) -> list[HumanMessage]:
        return [
            HumanMessage(
                content=f"""
                Use this content:\n,
                {handle_file_reading_for_model(rel_file.file_path)}
                It's concise description {rel_file.description}
                """
                + "\n".join(
                    [
                        f"Use it in section {rel_file_application.stage_name}, this document should be used in this section to {rel_file_application.application_desc}"
                        for rel_file_application in rel_file.application
                    ]
                )
            )
            for rel_file in state.relevant_files
        ]


class TopicSentenceManualReview(State):
    def __init__(self, model):
        super().__init__(model, "topic_sentence_manual_review")

    def run(self, state: AgentState, config: dict) -> dict:
        """
        Performs a manual review of the plan stage.

        :param state: current state of agent.
        :return: fields 'instruction' and 'plan' added to state.
        """

        logging.info(f"state {self.name}: running")
        review_topic_sentences = state.review_topic_sentences
        messages = state.messages
        instruction = config["configurable"]["instruction"]
        plan = state.plan
        if instruction:
            review_topic_sentences.append(instruction)
            messages.extend(
                [
                    HumanMessage(
                        content=(
                            TOPIC_SENTENCE_REVIEW_PROMPT + "\n\n"
                            f"Here is my task:\n\n{state.task}\n\n"
                            f"Here is my plan:\n\n{state.plan}\n\n"
                            f"Here is my instruction:\n\n{instruction}\n\n"
                            "Only return the Markdown for the new plan as output. "
                        )
                    )
                ]
            )
            response = self.model.invoke(messages)
            plan = response.content.strip()
            if "```markdown" in plan:
                plan = "\n".join(plan.split("\n")[1:-1])
            # sometimes, the LLM just decide it will not obey the instructions to not
            # add references. So, we will remove them here.
            plan = plan.strip()
            search = re.search(r"## References", plan)
            if search:
                _, r = search.span()
                plan = plan[:r]
            messages.append(AIMessage(content=plan))
        return {
            "state": self.name,
            "review_topic_sentences": review_topic_sentences,
            "plan": plan,
            "draft": plan,
            "messages": messages,
        }


class PaperWriter(State):
    def __init__(self, model):
        super().__init__(model, "paper_writer")

    def run(self, state: AgentState):
        """
        Generate the full draft of the paper based on the content, task and the plan.
        :param state: current state of the agent.
        :return: field 'draft' and 'revision_number' added to the paper.
        """

        logging.info(f"state {self.name}: running")
        content = "\n\n".join(state.content)
        critique = state.critique
        review_instructions = state.review_instructions
        task = state.task
        sentences_per_paragraph = state.sentences_per_paragraph
        # if previous state is internet_search, draft are in the form of topic senteces
        if state.state == "internet_search":
            additional_info = " in terms of topic sentences"
        else:
            additional_info = ""
        human_content = (
            "Generate a new draft of the document based on the "
            "information I gave you.\n\n"
            f"Here is my current draft{additional_info}:\n\n"
            f"{state.draft}\n\n"
        )
        messages = [
            SystemMessage(
                content=PAPER_WRITER_PROMPT.format(
                    task=task,
                    content=content,
                    review_instructions=review_instructions,
                    critique=critique,
                    sentences_per_paragraph=sentences_per_paragraph,
                )
            ),
            HumanMessage(content=human_content),
        ]
        response = self.model.invoke(messages)
        draft = response.content.strip()
        if "```markdown" in draft:
            draft = "\n".join(draft.split("\n")[1:-1])
        draft = draft.strip()
        search = re.search(r"## References", draft)
        if search:
            _, r = search.span()
            draft = draft[:r]
        return {
            "state": self.name,
            "draft": draft,
            "revision_number": state.revision_number + 1,
        }


class WriterManualReviewer(State):
    def __init__(self, model):
        super().__init__(model, "writer_manual_reviewer")

    def run(self, state: AgentState, config: dict) -> dict:
        """
        Performs manual review of the generated paper.
        :param state: current state of the paper.
        :return: Reviewed 'draft' and add to list of instructions.
        """

        logging.info(f"state {self.name}: running")
        review_instructions = state.review_instructions
        instruction = config["configurable"]["instruction"]
        draft = state.draft
        if instruction:
            review_instructions.append(instruction)
            joined_instructions = "\n".join(review_instructions)
            messages = [
                SystemMessage(content=WRITER_REVIEW_PROMPT),
                HumanMessage(
                    content=(
                        "Here is my task:\n\n"
                        f"{state.task}"
                        "\n\n"
                        "Here is my draft:\n\n"
                        f"{state.draft}"
                        "\n\n"
                        "Here is my instruction:\n\n"
                        f"{instruction}"
                        "\n\n"
                        "Here is my previous instructions that you must "
                        "observe:\n\n"
                        f"{joined_instructions}"
                        "\n\n"
                        "Only change in the draft what the user has requested by "
                        "the instruction.\n"
                        "Only return the Markdown for the new plan as output. "
                    )
                ),
            ]
            response = self.model.invoke(messages)
            draft = response.content.strip()
            if "```markdown" in draft:
                draft = "\n".join(draft.split("\n")[1:-1])
            search = re.search(r"## References", draft)
            if search:
                _, r = search.span()
                draft = draft[:r]
        return {
            "state": self.name,
            "review_instructions": review_instructions,
            "draft": draft,
        }


class ReflectionReviewer(State):
    def __init__(self, model):
        super().__init__(model, "reflection_reviewer")

    def run(self, state: AgentState) -> dict:
        """
        Performs reflection of the paper.
        :param state: current state of the agent.
        :return: 'critique' of the paper.
        """

        logging.info(f"state {self.name}: running")
        review_instructions = "\n".join(state.review_instructions)
        messages = [
            SystemMessage(
                content=REFLECTION_REVIEWER_PROMPT.format(
                    hypothesis=state.hypothesis,
                    review_instructions=review_instructions,
                )
            ),
            HumanMessage(content=state.draft),
        ]
        response = self.model.invoke(messages)
        return {"state": self.name, "critique": response.content}


class ReflectionManualReview(State):
    def __init__(self, model):
        super().__init__(model, "additional_reflection_instructions")

    def run(self, state: AgentState, config: dict) -> dict:
        """
        adds additional manual reflection for the review of the paper.
        :param state: current state of the agent.
        :param config: configuration with instruction.
        :return: 'critique' of the paper.
        """
        additional_critique = config["configurable"]["instruction"]
        critique = state.critique
        if additional_critique:
            critique = (
                critique + f"\n\nAdditional User's feedback:\n{additional_critique}\n"
            )
        return {"state": self.name, "critique": critique}


class ReflectionCritiqueReviewer(State):
    def __init__(self, model):
        super().__init__(model, "reflection_critique_reviewer")

    def run(self, state: AgentState):
        """
        Adds additional content to the reviewed paper.
        :param state: current state of the agent.
        :return: additional searched content to improve paper.
        """

        logging.info(f"state {self.name}: running")
        queries = {"queries": []}
        result = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        RESEARCH_CRITIQUE_PROMPT
                        + " You must only output the response in the"
                        + "JSON format '{ \"queries\": list[str] }' and no other text."
                    )
                ),
                HumanMessage(content=state.critique),
            ]
        ).content
        # we need to add this because sometimes the LLM decides to put a header
        # in the json file.
        if result[:7] == "```json":
            result = result.split("\n")
            result = "\n".join(result[1:-1])
        try:
            queries = json.loads(result)
        except:
            logging.warning(f"state {self.name}: could not extract query {result}.")
        content = state.content
        if queries["queries"]:
            search, cache = search_query_ideas(query_ideas=queries, cache=state.cache)
            content = content + search
        else:
            cache = state.cache
        return {"state": self.name, "cache": cache, "content": content}


class WriteAbstract(State):
    def __init__(self, model):
        super().__init__(model, "write_abstract")

    def run(self, state: AgentState):
        """
        Generate the abstract of the paper based on the draft, task and the plan.
        :param state: current state of the agent.
        :return: updated field 'draft' of the paper.
        """

        logging.info(f"state {self.name}: running")
        human_content = (
            f"Here is my task:\n\n{state.task}\n\n"
            f"Here is my plan:\n\n{state.plan}\n\n"
            f"Here is my research content:\n\n{state.content}"
            f"Here is my current draft:\n\n{state.draft}\n\n"
        )
        messages = [
            SystemMessage(content=ABSTRACT_WRITER_PROMPT),
            HumanMessage(content=human_content),
        ]
        response = self.model.invoke(messages)
        draft = response.content.strip()
        if "```markdown" in draft:
            draft = "\n".join(draft.split("\n")[1:-1])
        return {"state": self.name, "draft": draft}


class GenerateFigureCaptions(State):
    def __init__(self, model):
        super().__init__(model, "generate_figure_captions")

    def run(self, state: AgentState):
        """
        Generate figure captions.
        :param state: current state of the agent.
        :return: field 'draft' reviewed to the paper.
        """

        logging.info(f"state {self.name}: running")
        draft = state.draft
        pattern = r"!\[([^\]]*)\]\(([^\)]*)\)"

        # find all ![caption](file) definition of figures in markdown
        result = list(reversed(list(re.finditer(pattern, draft))))
        fig = len(result)

        # we process the figure list in reverse order because we change
        # the file
        for entry in result:
            left, right = entry.span()
            caption = (
                f'![]({entry[2]})\n\n<div align="center">Figure {fig}:'
                f"{entry[1]}</div>\n"
            )
            draft = draft[:left] + caption + draft[right:]
            fig -= 1

        return {"state": self.name, "draft": draft}


class GenerateReferences(State):
    def __init__(self, model):
        super().__init__(model, "generate_references")

    def run(self, state: AgentState):
        """
        Generate and references for the full draft of the paper.
        :param state: current state of the agent.
        :return: field 'references' reviewed to the paper.
        """

        logging.info(f"state {self.name}: running")
        content = state.content
        joined_content = "\n\n".join(content)
        human_content = (
            "Generate references for the following content entries. "
            "\n\n"
            "Content:"
            "\n\n"
            f"{joined_content}"
        )
        messages = [
            SystemMessage(content=REFERENCES_PROMPT),
            HumanMessage(content=human_content),
        ]
        response = self.model.invoke(messages)
        references = response.content.strip()
        if "```markdown" in references:
            references = "\n".join(references.split("\n")[1:-1])
        return {"state": self.name, "references": references}


class GenerateCitations(State):
    def __init__(self, model):
        super().__init__(model, "generate_citations")

    def run(self, state: AgentState):
        """
        Cite references in the paper.
        :param state: current state of the agent.
        :return: field 'draft' reviewed to the paper.
        """

        logging.info(f"state {self.name}: running")
        references = state.references
        draft = state.draft
        draft = draft + "\n\n" + references
        draft = insert_references(draft)
        return {"state": self.name, "draft": draft}


class LaTeXConverter(State):
    def __init__(self, model: ChatOpenAI):
        super().__init__(model, "latex_converter")

    def run(self, state: AgentState):
        """
        Convert LaTeX in the paper to Markdown format.
        :param state: current state of the agent.
        :return: field 'draft' reviewed to the paper.
        """

        logging.info(f"state {self.name}: running")
        logging.info(state)

        latex_draft = self._pandoc_convertion(state)

        logging.info(latex_draft)

        result = self.model.invoke(
            [
                SystemMessage(content=(LATEX_CONVERSION_PROMPT)),
                HumanMessage(
                    content=f"Refine the following LaTeX document:\n\n{latex_draft} to fit {state.latex_template} style, here is the official LaTeX template: \n{self._find_latex_template(state.latex_template)}"
                ),
            ]
        )
        logging.info(result)

        return {"state": self.name, "latex_draft": result.content}

    def _find_latex_template(self, latex_template: str) -> str:
        try:
            logging.info(f"{Path.cwd()}, {latex_template.lower()}.tex")

            latex_template_path = next(
                iter(Path.cwd().rglob(f"*{latex_template.lower()}.tex"))
            )
            with latex_template_path.open("r", encoding="utf-8") as f:
                return f.read()
        except Exception as err:
            logging.error(f"Error occured while looking for latex template: {err}")
            return "Use standard LaTeX template"

    def _pandoc_convertion(self, state: AgentState) -> str:
        tmp_file = "/tmp" / Path(f"{state.title}_{uuid.uuid4()}.md")

        with tmp_file.open(mode="w", encoding="utf-8") as f:
            f.write(state.draft)

        pandoc_run = subprocess.run(
            [
                "pandoc",
                "-s",
                str(tmp_file),
                "-f",
                "markdown",
                "-t",
                "latex",
                "-o",
                str(tmp_file.with_suffix(".tex")),
            ],
            capture_output=True,
            text=True,
        )

        logging.info(pandoc_run)

        with (tmp_file.with_suffix(".tex")).open(mode="r", encoding="utf-8") as f:
            return f.read()
