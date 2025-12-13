# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import json
import logging
import os
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set

from decouple import config
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agents.utils import create_log_entry
from src.utils.constants import SUPPORTED_TABLE_FORMATS
from src.utils.file_handlers import extract_table_data, handle_file_reading_for_model
from src.utils.models import (
    AgentState,
    GeneratedPlotImage,
    PlotSuggestion,
    RelevantaFileApplication,
    RelevantFile,
    WorkflowLog,
)

from .gen_citations import insert_references
from .prompts import (
    ABSTRACT_WRITER_PROMPT,
    INTERNET_SEARCH_PROMPT,
    LATEX_CONVERSION_PROMPT,
    PAPER_WRITER_PROMPT,
    PLOT_DATA_CONTEXT_FILE_TEMPLATE,
    PLOT_DATA_CONTEXT_INLINE_TEMPLATE,
    PLOT_SUGGESTION_PROMPT,
    PLOT_SUGGESTION_REQUEST_TEMPLATE,
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
        self.name = f"{name}_graph_state"


class AnalyzeRelevantFiles(State):
    def __init__(state, model: ChatOpenAI) -> None:
        super().__init__(model, "analyze_relevant_files")

    def run(self, state: AgentState, config: dict) -> dict:
        logging.info(f"state {self.name}: running")

        def process_file(
            relevant_file: RelevantFile,
        ) -> tuple[RelevantFile, list[WorkflowLog]]:
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

            local_logs = [
                create_log_entry(msg, "analyze_relevant_files") for msg in messages
            ]

            local_logs.append(create_log_entry(response, "analyze_relevant_files"))  # type: ignore

            json_response = json.loads(response.content)  # type: ignore
            for section_name, application in json_response.items():
                relevant_file.application.append(
                    RelevantaFileApplication(
                        stage_name=section_name, application_desc=application
                    )
                )
            return relevant_file, local_logs

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_file, relevant_file)
                for relevant_file in state.relevant_files
            ]

            results = [future.result() for future in as_completed(futures)]
            relevant_files = [r[0] for r in results]
            new_logs = [log for r in results for log in r[1]]

            return {
                "relevant_files": relevant_files,
                "workflow_logs": state.workflow_logs + new_logs,
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

        logs = []

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

        logs.extend(
            [create_log_entry(message, "suggest_title") for message in messages]
        )

        return {
            "state": self.name,
            "title": response.content,
            "messages": messages,
            "draft": response.content,
            "workflow_logs": state.workflow_logs + logs,
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

        logs = []
        messages = state.messages
        instruction = config["configurable"]["instruction"]
        if not instruction:
            human_message = HumanMessage(
                content="Just return the final title without any additional information"
            )
        else:
            human_message = HumanMessage(content=instruction)
        messages.append(human_message)
        logs.append(create_log_entry(human_message, "suggest_title"))

        response = self.model.invoke(messages)
        logs.append(create_log_entry(response, "suggest_title"))

        messages.append(response)
        if not instruction:
            messages = []
        title = response.content
        return {
            "state": self.name,
            "title": title,
            "messages": messages,
            "draft": response.content,
            "workflow_logs": state.workflow_logs + logs,
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
        logs = []

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
            messages = [
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
            ]
            logs.extend([create_log_entry(m, "internet_search") for m in messages])

            response = self.model.invoke(
                messages, response_format={"type": "json_object"}
            )
            logs.append(create_log_entry(response, "internet_search"))

            result = response.content
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
            "workflow_logs": state.workflow_logs + logs,
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
        logs = []
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

        logs.extend([create_log_entry(m, "topic_sentence_writer") for m in messages])

        response = self.model.invoke(messages)
        logs.append(create_log_entry(response, "topic_sentence_writer"))

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
            "plan": plan,
            "draft": plan,
            "messages": messages,
            "workflow_logs": state.workflow_logs + logs,
        }

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
        logs = []
        review_topic_sentences = state.review_topic_sentences
        messages = state.messages
        instruction = config["configurable"]["instruction"]
        plan = state.plan
        if instruction:
            review_topic_sentences.append(instruction)
            new_messages = [
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
            messages.extend(new_messages)
            logs.extend(
                [
                    create_log_entry(m, "topic_sentence_manual_review")
                    for m in new_messages
                ]
            )

            response = self.model.invoke(messages)
            logs.append(create_log_entry(response, "topic_sentence_manual_review"))

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
            "workflow_logs": state.workflow_logs + logs,
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
        logs = []
        content = "\n\n".join(state.content or [])
        critique = state.critique
        review_instructions = state.review_instructions
        task = state.task
        sentences_per_paragraph = state.sentences_per_paragraph

        images_info = self._get_available_images(state)
        if images_info:
            content += "\n\nAvailable Images:\n" + images_info

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

        if images_info:
            human_content += f"\nIMPORTANT: Include these available images in your document:\n{images_info}\n"
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
        logs.extend([create_log_entry(m, "paper_writer") for m in messages])

        response = self.model.invoke(messages)
        logs.append(create_log_entry(response, "paper_writer"))

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
            "revision_number": (state.revision_number or 0) + 1,
            "generated_plot_images": state.generated_plot_images or [],
            "workflow_logs": state.workflow_logs + logs,
        }

    def _get_available_images(self, state: AgentState):
        """Collect information about available images for inclusion in the document."""
        images_info = []
        generated_images = [
            GeneratedPlotImage.model_validate(img)
            for img in (state.generated_plot_images or [])
        ]
        logging.info(f"Found {len(generated_images)} generated plot images")
        for img_info in generated_images:
            description = img_info.description or "Generated plot"
            absolute_path = img_info.path or ""

            if absolute_path and os.path.exists(absolute_path):
                filename = os.path.basename(absolute_path)
                relative_path = f"../images/{filename}"
                images_info.append(f"- ![{description}]({relative_path})")
                logging.info(f"Added generated image: {description} -> {relative_path}")
            else:
                fallback_path = img_info.relative_path or ""
                if fallback_path:
                    images_info.append(f"- ![{description}]({fallback_path})")
                    logging.info(
                        f"Added generated image (fallback): {description} -> {fallback_path}"
                    )

        # Use KIROKU_PROJECT_DIRECTORY environment variable or fallback to hardcoded path
        working_dir = os.environ.get(
            "KIROKU_PROJECT_DIRECTORY",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "proj"
            ),
        )
        images_dir = os.path.join(working_dir, "images")
        logging.info(f"Looking for images in: {images_dir}")

        if os.path.exists(images_dir):
            filenames = os.listdir(images_dir)
            logging.info(f"Found files in images directory: {filenames}")

            existing_filenames: set[str] = set(
                Path(img.path).name for img in generated_images if img.path
            )
            existing_filenames.update(
                Path(img.relative_path).name
                for img in generated_images
                if img.relative_path
            )

            for filename in filenames:
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
                    if filename in existing_filenames:
                        continue

                    name = (
                        os.path.splitext(filename)[0]
                        .replace("-", " ")
                        .replace("_", " ")
                        .title()
                    )
                    relative_path = f"../images/{filename}"
                    images_info.append(f"- ![{name}]({relative_path})")
                    logging.info(f"Added existing image: {name} -> {relative_path}")
        else:
            logging.warning(f"Images directory does not exist: {images_dir}")

        logging.info(f"Total images found: {len(images_info)}")
        result = "\n".join(images_info) if images_info else ""
        logging.info(f"Images info to be added to content: {result}")
        return result


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
        logs = []
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
            logs.extend(
                [create_log_entry(m, "writer_manual_reviewer") for m in messages]
            )
            response = self.model.invoke(messages)
            logs.append(create_log_entry(response, "writer_manual_reviewer"))
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
            "generated_plot_images": state.generated_plot_images or [],
            "workflow_logs": state.workflow_logs + logs,
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
        logs = []
        review_instructions = "\n".join(state.review_instructions)
        messages = [
            SystemMessage(
                content=REFLECTION_REVIEWER_PROMPT.format(
                    hypothesis=state.hypothesis, review_instructions=review_instructions
                )
            ),
            HumanMessage(content=state.draft),
        ]
        logs.extend([create_log_entry(m, "reflection_reviewer") for m in messages])
        response = self.model.invoke(messages)
        logs.append(create_log_entry(response, "reflection_reviewer"))
        return {
            "state": self.name,
            "critique": response.content,
            "workflow_logs": state.workflow_logs + logs,
        }


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
        logging.info(f"state {self.name}: running")
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
        logs = []
        queries = {"queries": []}
        messages = [
            SystemMessage(
                content=(
                    RESEARCH_CRITIQUE_PROMPT
                    + " You must only output the response in the"
                    + "JSON format '{ \"queries\": list[str] }' and no other text."
                )
            ),
            HumanMessage(content=state.critique),
        ]
        logs.extend(
            [create_log_entry(m, "reflection_critique_reviewer") for m in messages]
        )
        response = self.model.invoke(messages)
        logs.append(create_log_entry(response, "reflection_critique_reviewer"))
        result = response.content
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
        return {
            "state": self.name,
            "cache": cache,
            "content": content,
            "workflow_logs": state.workflow_logs + logs,
        }


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
        logs = []
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
        logs.extend([create_log_entry(m, "write_abstract") for m in messages])
        response = self.model.invoke(messages)
        logs.append(create_log_entry(response, "write_abstract"))
        draft = response.content.strip()
        if "```markdown" in draft:
            draft = "\n".join(draft.split("\n")[1:-1])
        return {
            "state": self.name,
            "draft": draft,
            "generated_plot_images": state.generated_plot_images or [],
            "workflow_logs": state.workflow_logs + logs,
        }


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
            image_path = entry[2]
            image_caption = entry[1] if entry[1] else "Image"
            caption = (
                f'![{image_caption}]({image_path})\n\n<div align="center">Figure {fig}: '
                f"{image_caption}</div>\n"
            )
            draft = draft[:left] + caption + draft[right:]
            fig -= 1

        return {
            "state": self.name,
            "draft": draft,
            "generated_plot_images": state.generated_plot_images
            or [],  # Preserve plot images
        }


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
        logs = []
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
        logs.extend([create_log_entry(m, "generate_references") for m in messages])
        response = self.model.invoke(messages)
        logs.append(create_log_entry(response, "generate_references"))
        references = response.content.strip()
        if "```markdown" in references:
            references = "\n".join(references.split("\n")[1:-1])
        return {
            "state": self.name,
            "references": references,
            "workflow_logs": state.workflow_logs + logs,
        }


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
        return {
            "state": self.name,
            "draft": draft,
            "generated_plot_images": state.generated_plot_images or [],
        }


class LaTeXConverter(State):
    def __init__(self, model: ChatOpenAI) -> None:
        super().__init__(model, "latex_converter")

    def run(self, state: AgentState) -> dict:
        """
        Convert LaTeX in the paper to Markdown format.
        :param state: current state of the agent.
        :return: field 'draft' reviewed to the paper.
        """

        logging.info(f"state {self.name}: running")
        logs = []

        latex_draft = self._pandoc_convertion(state)

        logging.info(latex_draft)

        messages = [
            SystemMessage(content=(LATEX_CONVERSION_PROMPT)),
            HumanMessage(
                content=f"Refine the following LaTeX document:\n\n{latex_draft} to fit {state.latex_template} style, here is the official LaTeX template: \n{self._find_latex_template(state.latex_template)}"
            ),
        ]
        logs.extend([create_log_entry(m, "latex_converter") for m in messages])
        result = self.model.invoke(messages)
        logs.append(create_log_entry(result, "latex_converter"))
        logging.info(result)

        return {
            "state": self.name,
            "latex_draft": result.content,
            "workflow_logs": state.workflow_logs + logs,
        }

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


class PlotSuggestionAgent(State):
    def __init__(self, model: ChatOpenAI):
        super().__init__(model, "plot_suggestion")

    def run(self, state: AgentState) -> dict:
        """
        Generate plot suggestions based on the paper plan and content.
        """
        logging.info(f"state {self.name}: running")
        logs = []

        plan = state.plan
        task = state.task
        content = "\n\n".join(state.content or [])

        plot_suggestions = {}

        plot_data = state.plot_data

        for i, relevant_file in enumerate(state.relevant_files):
            if relevant_file.file_path.suffix in SUPPORTED_TABLE_FORMATS:
                plot_data = relevant_file.file_path

                logging.info(f"Processing file under path: {plot_data}")

                data_preview, columns_desc, shape = extract_table_data(
                    relevant_file.file_path
                )

                data_context = PLOT_DATA_CONTEXT_FILE_TEMPLATE.format(
                    plot_path=str(plot_data),
                    data_preview=data_preview,
                    columns=columns_desc,
                    shape=shape,
                    filename=plot_data.name,
                )
                section_application_map = "\n".join(
                    [
                        f"- Use it in section {rel_file_application.stage_name}, this document should be used in this section to {rel_file_application.application_desc}"
                        for rel_file_application in relevant_file.application
                    ]
                )

                prompt = PLOT_SUGGESTION_REQUEST_TEMPLATE.format(
                    plan=plan,
                    content_preview=content[:2000],
                    data_context=data_context,
                    description_of_data=relevant_file.description,
                    section_application_mapping=section_application_map,
                    task=task,
                )

                messages = [
                    SystemMessage(content=PLOT_SUGGESTION_PROMPT),
                    HumanMessage(content=prompt),
                ]
                logs.extend([create_log_entry(m, "plot_suggestion") for m in messages])

                response = self.model.invoke(messages)
                logging.info(f"Suggestiong plot N{i + 1}: {response.content[:100]}")
                logs.append(create_log_entry(response, "plot_suggestion"))

                plot_suggestions[i] = response.content.strip()

        data_context = ""

        try:
            import uuid
            import re

            suggested_plots: list[PlotSuggestion] = []

            for i, response_text in plot_suggestions.items():
                plot_sections = re.split(r"\bPLOT\s+\d+:", response_text)

                for j, section in enumerate(plot_sections[1:], 1):
                    desc_match = re.search(
                        r"Description:\s*(.+?)(?=\n\w+:|$)", section, re.DOTALL
                    )
                    description = (
                        desc_match.group(1).strip()
                        if desc_match
                        else f"Visualization {j}"
                    )

                    rationale_match = re.search(
                        r"Rationale:\s*(.+?)(?=\n\w+:|$)", section, re.DOTALL
                    )
                    rationale = (
                        rationale_match.group(1).strip()
                        if rationale_match
                        else "No rationale provided"
                    )

                    code_match = re.search(
                        r"```python\s*\n(.*?)\n```", section, re.DOTALL
                    )
                    code = code_match.group(1).strip() if code_match else ""

                    logging.info("Suggestion N {i} section {j} successfully parsed.")

                    if description or code:  # Only add if we have some content
                        suggested_plots.append(
                            PlotSuggestion(
                                id=str(uuid.uuid4()),
                                description=description,
                                code=code,
                                rationale=rationale,
                                approved=False,
                                filename_base=state.relevant_files[i].file_path,
                            )
                        )

        except Exception as e:
            logging.error(f"Failed to parse plot suggestions: {e}")
            # Fallback: create a simple plot suggestion
            suggested_plots = [
                PlotSuggestion(
                    id=str(uuid.uuid4()),
                    description="Sample visualization",
                    code=(
                        "import matplotlib.pyplot as plt\nimport numpy as np\n\n"
                        "x = np.linspace(0, 10, 100)\ny = np.sin(x)\n\n"
                        "plt.figure(figsize=(8, 6))\nplt.plot(x, y)\n"
                        "plt.title('Sample Plot')\nplt.xlabel('X values')\nplt.ylabel('Y values')\nplt.show()"
                    ),
                    rationale="A basic visualization to demonstrate the concept",
                    approved=False,
                )
            ]

        return {
            "state": self.name,
            "suggested_plots": suggested_plots,
            "draft": plan,  # Keep the plan as draft for display
            "workflow_logs": state.workflow_logs + logs,
        }


class PlotApprovalAgent(State):
    def __init__(self, model: ChatOpenAI):
        super().__init__(model, "plot_approval")

    def run(self, state: AgentState, config: dict) -> dict:
        """
        Handle plot approval/rejection based on user input.
        """
        logging.info(f"state {self.name}: running")

        suggested_plots = [
            PlotSuggestion.model_validate(p) for p in state.suggested_plots
        ]
        instruction = config["configurable"].get("instruction", "")

        logging.info(f"PlotApprovalAgent received instruction: '{instruction}'")

        if instruction:
            if "apply individual selections" in instruction.lower():
                logging.info(
                    "Processing checkbox selections - approval status already updated"
                )
            elif "approve" in instruction.lower():
                if "all" in instruction.lower():
                    for plot in suggested_plots:
                        plot.approved = True
                    logging.info("Approved all plots")
                else:
                    plot_ids = [
                        s.strip()
                        for s in instruction.lower()
                        .replace("approve", "")
                        .strip()
                        .split(",")
                    ]
                    for plot_id in plot_ids:
                        try:
                            idx = int(plot_id) - 1
                            if 0 <= idx < len(suggested_plots):
                                suggested_plots[idx].approved = True
                                logging.info(f"Approved plot {idx + 1}")
                        except (ValueError, IndexError):
                            pass
            elif "reject" in instruction.lower():
                if "all" in instruction.lower():
                    for plot in suggested_plots:
                        plot.approved = False
                    logging.info("Rejected all plots")
                else:
                    plot_ids = [
                        s.strip()
                        for s in instruction.lower()
                        .replace("reject", "")
                        .strip()
                        .split(",")
                    ]
                    for plot_id in plot_ids:
                        try:
                            idx = int(plot_id) - 1
                            if 0 <= idx < len(suggested_plots):
                                suggested_plots[idx].approved = False
                                logging.info(f"Rejected plot {idx + 1}")
                        except (ValueError, IndexError):
                            pass

        return {
            "state": self.name,
            "suggested_plots": suggested_plots,
            "draft": self._format_plot_summary(suggested_plots),
        }

    def _format_plot_summary(self, plots: list[PlotSuggestion]) -> str:
        """Format plots for display in the UI."""
        if not plots:
            return "No plots suggested."

        summary = "## Suggested Plots\n\n"
        for i, plot in enumerate(plots, 1):
            status = "✅ APPROVED" if plot.approved else "❌ Pending"
            summary += f"### Plot {i}: {plot.description or 'Untitled'} [{status}]\n\n"
            summary += f"**Rationale:** {plot.rationale or 'No rationale provided'}\n\n"
            summary += (
                f"```python\n{(plot.code or 'No code provided')[:300]}...\n```\n\n"
            )

        summary += "\n**Instructions:** Type 'approve 1,3' to approve plots 1 and 3, or 'reject 2' to reject plot 2."
        return summary


class PlotGenerationAgent(State):
    def __init__(self, model: ChatOpenAI):
        super().__init__(model, "plot_generation")

    def run(self, state: AgentState) -> dict:
        """
        Execute approved plots via CodeAPI and save images.
        """
        logging.info(f"state {self.name}: running")

        from pathlib import Path
        from src.utils.codeapi import CodeAPI
        import os

        suggested_plots = [
            PlotSuggestion.model_validate(p) for p in state.suggested_plots
        ]
        plot_data = state.plot_data
        generated_images: list[GeneratedPlotImage] = []

        # Initialize CodeAPI if available
        codeapi_url = config("CODEAPI_URL", "https://api.codapi.org/v1/exec")
        logging.info(f"CODEAPI_URL: {codeapi_url}")
        if not codeapi_url:
            logging.error("CODEAPI_URL not configured, skipping plot generation")
            logging.error(
                "To enable plot generation, set environment variable: export CODEAPI_URL=http://your-codeapi-server"
            )
            return {
                "state": self.name,
                "generated_plot_images": [],
                "draft": "Plot generation skipped: CODEAPI_URL not configured. Set CODEAPI_URL environment variable to enable plot generation.",
            }

        try:
            codeapi = CodeAPI(codeapi_url)
            logging.info(f"Successfully initialized CodeAPI with URL: {codeapi_url}")
        except Exception as e:
            logging.error(f"Failed to initialize CodeAPI: {e}")
            return {
                "state": self.name,
                "generated_plot_images": [],
                "draft": f"Plot generation failed: Could not connect to CodeAPI at {codeapi_url}. Error: {e}",
            }

        # Set up capture epilogue for plot extraction
        capture_code = """
# --- kiroku: capture figures to base64 stdout ---
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as _plt
    import io as _io, base64 as _b64
    
    # Ensure we have at least one figure
    figs = _plt.get_fignums()
    if not figs:
        print("No figures found, creating default figure")
        _plt.figure(figsize=(8, 6))
        _plt.text(0.5, 0.5, 'Plot generation failed', ha='center', va='center')
        _plt.title('Error: No plot generated')
        figs = _plt.get_fignums()
    
    print(f"Found {len(figs)} figure(s)")
    for _i in figs:
        _fig = _plt.figure(_i)
        _buf = _io.BytesIO()
        _fig.savefig(_buf, format='png', bbox_inches='tight', dpi=100)
        _buf.seek(0)
        _enc = _b64.b64encode(_buf.getvalue()).decode('ascii')
        print('data:image/png;base64,' + _enc)
        print(f"Generated image for figure {_i}")
        
except Exception as _e:
    print(f'capture_error: {_e}')
    import traceback
    print(f'capture_traceback: {traceback.format_exc()}')
"""

        # Process approved plots
        for i, plot in enumerate(suggested_plots):
            if not plot.approved:
                continue

            plot_code = plot.code or ""
            if not plot_code:
                continue

            # Prepare matplotlib setup and code
            matplotlib_setup = """
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""

            # Prepare data loading code and files for CodeAPI
            files = {}
            data_bootstrap = ""

            plot_data = plot.filename_base
            if plot_data:
                try:
                    import os
                    from io import StringIO

                    import pandas as pd

                    # Check if plot_data is a file path
                    if plot_data.is_file():
                        # Read the CSV file and pass it to CodeAPI
                        # TODO proper handling of different file types
                        with plot_data.open("r", encoding="utf-8") as f:
                            csv_content = f.read()

                        filename = plot_data.name
                        files[filename] = csv_content

                        data_bootstrap = f"""
# Load data from uploaded file
import pandas as pd
data = pd.read_csv('{filename}')
"""
                    else:
                        files["data.csv"] = plot_data
                        data_bootstrap = """
# Load data from uploaded CSV
import pandas as pd
data = pd.read_csv('data.csv')
"""
                        logging.info(
                            "Prepared inline CSV data as data.csv for CodeAPI upload"
                        )

                except Exception as e:
                    logging.error(f"Failed to prepare plot data: {e}")
                    data_bootstrap = "# No data available\n"

            full_code = (
                matplotlib_setup
                + "\n"
                + data_bootstrap
                + "\n"
                + plot_code
                + "\n"
                + capture_code
            )

            try:
                logging.info(f"Executing plot {i + 1}: {plot.description or 'Unknown'}")
                logging.debug(f"Plot code length: {len(full_code)} characters")
                if files:
                    logging.info(
                        f"Uploading {len(files)} files to CodeAPI: {list(files.keys())}"
                    )

                # Execute via CodeAPI with files
                result = codeapi.run_python(full_code, inputs=files)
                logging.info(
                    f"CodeAPI execution result for plot {i + 1}: {type(result)}"
                )

                images_b64 = CodeAPI.extract_images_base64(result)

                if images_b64:
                    # Save images to the workspace
                    import time

                    working_dir = Path(os.environ.get("KIROKU_PROJECT_DIRECTORY", "."))
                    images_dir = working_dir / "images"
                    images_dir.mkdir(parents=True, exist_ok=True)

                    for j, b64_img in enumerate(images_b64):
                        img_bytes = CodeAPI.b64_to_bytes(b64_img)

                        # Use descriptive filename based on plot description
                        filename_base = plot.filename_base or f"plot_{i + 1}"
                        if len(images_b64) > 1:
                            filename = f"{filename_base}_{j + 1}.png"
                        else:
                            filename = f"{filename_base}.png"

                        img_path = images_dir / filename

                        with open(img_path, "wb") as f:
                            f.write(img_bytes)

                        # Store both the full path and the relative path for markdown
                        relative_path = f"images/{filename}"
                        generated_images.append(
                            GeneratedPlotImage(
                                path=str(img_path),
                                relative_path=relative_path,
                                description=plot.description or "Generated plot",
                                rationale=plot.rationale or "",
                            )
                        )
                        logging.info(f"Saved plot image: {img_path}")
                else:
                    logging.warning(f"No images generated for plot {i + 1}")

            except Exception as e:
                logging.error(f"Failed to execute plot {i + 1}: {e}")

        summary = (
            f"Generated {len(generated_images)} plot images from "
            f"{len([p for p in suggested_plots if p.approved])} approved plots."
        )

        return {
            "state": self.name,
            "generated_plot_images": generated_images,
            "draft": summary,
        }
