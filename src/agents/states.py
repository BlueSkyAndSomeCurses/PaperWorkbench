# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import json
import logging
import os
import re
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

from .gen_citations import insert_references
from .prompts import (
    ABSTRACT_WRITER_PROMPT,
    INTERNET_SEARCH_PROMPT,
    LATEX_CONVERSION_PROMPT,
    PAPER_WRITER_PROMPT,
    REFERENCES_PROMPT,
    REFLECTION_REVIEWER_PROMPT,
    RESEARCH_CRITIQUE_PROMPT,
    TASK_TEMPLATE,
    TITLE_PROMPT,
    TOPIC_SENTENCE_PROMPT,
    TOPIC_SENTENCE_REVIEW_PROMPT,
    WRITER_REVIEW_PROMPT,
)
from .search import *


class AgentState(TypedDict):
    state: str

    title: str
    messages: list
    hypothesis: str
    area_of_paper: str
    type_of_document: str
    section_names: str
    number_of_paragraphs: str
    results: str
    references: list[str]

    # these are instructions that we save for the topic sentences
    # and paper writing
    review_topic_sentences: list[str]
    review_instructions: list[str]

    task: str
    plan: str
    draft: str
    critique: str
    cache: set[str]
    content: list[str]
    revision_number: int
    number_of_queries: int
    max_revisions: int
    sentences_per_paragraph: int
    latex_draft: str

    suggested_plots: list[dict]
    plot_data: str
    generated_plot_images: list[str]


class State:
    def __init__(self, model, name):
        self.model = model
        self.name = name


class SuggestTitle(State):
    def __init__(self, model):
        super().__init__(model, "suggest_title")

    def run(self, state: AgentState, config: dict) -> dict:
        """Node of graph that suggests a title for the paper.

        :param state: state of the agent.
        :return: fields 'title', 'draft' and 'messages' updated for the paper.
        """

        logging.info(f"state {self.name}: running")

        messages = state["messages"]
        if not messages:
            title = state["title"]
            area_of_paper = state["area_of_paper"]
            hypothesis = state["hypothesis"]

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

        messages = state["messages"]
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
            ", ".join([f"'{section}'" for section in section_names[:-1]])
            + f" and '{section_names[-1]}'"
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
                    section_names, number_of_paragraphs
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

        queries = {"queries": []}
        task = self.create_task(
            title=state["title"],
            hypothesis=state["hypothesis"],
            area_of_paper=state["area_of_paper"],
            type_of_document=state["type_of_document"],
            section_names=state["section_names"],
            number_of_paragraphs=state["number_of_paragraphs"],
            results=state["results"],
            references=state["references"],
        )
        for _ in range(3):  # three attempts
            result = self.model.invoke(
                [
                    SystemMessage(
                        content=(
                            INTERNET_SEARCH_PROMPT.format(
                                number_of_queries=state["number_of_queries"]
                            )
                            + " You must only output the response in a plain list of queries "
                            "in the JSON format '{ \"queries\": list[str] }' and no other text. "
                            "You MUST only cite references that are in the references "
                            "section. "
                        )
                    ),
                    HumanMessage(content=task),
                ]
            ).content
            # we need to add this because sometimes the LLM decides to put a header
            # in the json file.
            if result[:7] == "```json":
                result = result.split("\n")
                result = "\n".join(result[1:-1])
            content = state.get("content", [])
            try:
                queries = json.loads(result)
                break
            except:
                logging.warning(f"state {self.name}: could not extract query {result}.")
        # finally, add to the queries all references that have http
        for ref in state["references"]:
            search_match = re.search(r"http.*(\s|$)", ref)
            if search_match:
                l, r = search_match.span()
                http_ref = ref[l:r]
                queries["queries"].insert(0, http_ref)
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
        task = state["task"]
        content = "\n\n".join(state["content"])
        messages = state["messages"]
        if not messages:
            messages = [SystemMessage(content=TOPIC_SENTENCE_PROMPT)]
        messages.append(
            HumanMessage(
                content=(
                    f"This is the content of a search on the internet for the paper:\n\n"
                    f"{content}\n\n"
                    f"{task}"
                )
            )
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
        return {"state": self.name, "plan": plan, "draft": plan, "messages": messages}


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
        review_topic_sentences = state.get("review_topic_sentences", [])
        messages = state["messages"]
        instruction = config["configurable"]["instruction"]
        plan = state["plan"]
        if instruction:
            review_topic_sentences.append(instruction)
            messages.extend(
                [
                    HumanMessage(
                        content=(
                            TOPIC_SENTENCE_REVIEW_PROMPT + "\n\n"
                            f"Here is my task:\n\n{state['task']}\n\n"
                            f"Here is my plan:\n\n{state['plan']}\n\n"
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
        content = "\n\n".join(state.get("content", []))
        critique = state.get("critique", "")
        review_instructions = state.get("review_instructions", [])
        task = state["task"]
        sentences_per_paragraph = state["sentences_per_paragraph"]

        images_info = self._get_available_images(state)
        if images_info:
            content += "\n\nAvailable Images:\n" + images_info
        
        # if previous state is internet_search, draft are in the form of topic senteces
        if state["state"] == "internet_search":
            additional_info = " in terms of topic sentences"
        else:
            additional_info = ""
        human_content = (
            "Generate a new draft of the document based on the "
            "information I gave you.\n\n"
            f"Here is my current draft{additional_info}:\n\n"
            f"{state['draft']}\n\n"
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
            "revision_number": state.get("revision_number", 1) + 1,
            "generated_plot_images": state.get("generated_plot_images", []),
        }
    
    def _get_available_images(self, state: AgentState):
        """Collect information about available images for inclusion in the document."""
        images_info = []

        generated_images = state.get("generated_plot_images", [])
        logging.info(f"Found {len(generated_images)} generated plot images")
        for img_info in generated_images:
            description = img_info.get("description", "Generated plot")
            absolute_path = img_info.get("path", "")
            
            if absolute_path and os.path.exists(absolute_path):
                filename = os.path.basename(absolute_path)
                relative_path = f"../images/{filename}"
                images_info.append(f"- ![{description}]({relative_path})")
                logging.info(f"Added generated image: {description} -> {relative_path}")
            else:
                fallback_path = img_info.get("relative_path", "")
                if fallback_path:
                    images_info.append(f"- ![{description}]({fallback_path})")
                    logging.info(f"Added generated image (fallback): {description} -> {fallback_path}")
        
        # Use KIROKU_PROJECT_DIRECTORY environment variable or fallback to hardcoded path
        working_dir = os.environ.get("KIROKU_PROJECT_DIRECTORY", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "proj"))
        images_dir = os.path.join(working_dir, "images")
        logging.info(f"Looking for images in: {images_dir}")
        
        if os.path.exists(images_dir):
            filenames = os.listdir(images_dir)
            logging.info(f"Found files in images directory: {filenames}")
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                    skip_file = False
                    for img_info in generated_images:
                        if (img_info.get("filename", "") == filename or 
                            img_info.get("relative_path", "").endswith(filename)):
                            skip_file = True
                            break
                    if not skip_file:
                        name = os.path.splitext(filename)[0].replace('-', ' ').replace('_', ' ').title()
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
        review_instructions = state.get("review_instructions", [])
        instruction = config["configurable"]["instruction"]
        draft = state["draft"]
        if instruction:
            review_instructions.append(instruction)
            joined_instructions = "\n".join(review_instructions)
            messages = [
                SystemMessage(content=WRITER_REVIEW_PROMPT),
                HumanMessage(
                    content=(
                        "Here is my task:\n\n"
                        f"{state['task']}"
                        "\n\n"
                        "Here is my draft:\n\n"
                        f"{state['draft']}"
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
            "generated_plot_images": state.get("generated_plot_images", []),
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
        review_instructions = "\n".join(state.get("review_instructions", []))
        messages = [
            SystemMessage(
                content=REFLECTION_REVIEWER_PROMPT.format(
                    hypothesis=state["hypothesis"],
                    review_instructions=review_instructions,
                )
            ),
            HumanMessage(content=state["draft"]),
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
        critique = state["critique"]
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
                HumanMessage(content=state["critique"]),
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
        content = state.get("content", [])
        if queries["queries"]:
            search, cache = search_query_ideas(
                query_ideas=queries, cache=state.get("cache", set())
            )
            content = content + search
        else:
            cache = state.get("cache", set())
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
            f"Here is my task:\n\n{state['task']}\n\n"
            f"Here is my plan:\n\n{state['plan']}\n\n"
            f"Here is my research content:\n\n{state['content']}"
            f"Here is my current draft:\n\n{state['draft']}\n\n"
        )
        messages = [
            SystemMessage(content=ABSTRACT_WRITER_PROMPT),
            HumanMessage(content=human_content),
        ]
        response = self.model.invoke(messages)
        draft = response.content.strip()
        if "```markdown" in draft:
            draft = "\n".join(draft.split("\n")[1:-1])
        return {
            "state": self.name, 
            "draft": draft,
            "generated_plot_images": state.get("generated_plot_images", []),
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
        draft = state["draft"]
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
            "generated_plot_images": state.get("generated_plot_images", []),  # Preserve plot images
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
        content = state["content"]
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
        references = state["references"]
        draft = state["draft"]
        draft = draft + "\n\n" + references
        draft = insert_references(draft)
        return {
            "state": self.name, 
            "draft": draft,
            "generated_plot_images": state.get("generated_plot_images", []),
        }


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
        draft = state["draft"]

        result = self.model.invoke(
            [
                SystemMessage(content=(LATEX_CONVERSION_PROMPT)),
                HumanMessage(
                    content=f"Convert the following markdown to LaTeX:\n\n{draft}"
                ),
            ]
        )
        logging.info(result)

        return {"state": self.name, "latex_draft": result.content}


class PlotSuggestionAgent(State):
    def __init__(self, model: ChatOpenAI):
        super().__init__(model, "plot_suggestion")

    def run(self, state: AgentState) -> dict:
        """
        Generate plot suggestions based on the paper plan and content.
        """
        logging.info(f"state {self.name}: running")
        
        plan = state.get("plan", "")
        task = state.get("task", "")
        content = "\n\n".join(state.get("content", []))
        plot_data = state.get("plot_data", "")

        data_context = ""
        if plot_data:
            try:
                import pandas as pd
                import os
                from io import StringIO
                
                if os.path.isfile(plot_data):
                    df = pd.read_csv(plot_data)
                    data_preview = df.head().to_string()
                    filename = os.path.basename(plot_data)
                    data_context = f"""
Available Data (from {plot_data}):
{data_preview}

Data columns: {list(df.columns)}
Data shape: {df.shape}

Note: The CSV file will be available as '{filename}' and also loaded as variable 'data' in your plotting code.
You can use: data = pd.read_csv('{filename}')
"""
                else:
                    df = pd.read_csv(StringIO(plot_data))
                    data_preview = df.head().to_string()
                    data_context = f"""
Available Data:
{data_preview}

Data columns: {list(df.columns)}
Data shape: {df.shape}

Note: The CSV data will be available as 'data.csv' and also loaded as variable 'data' in your plotting code.
You can use: data = pd.read_csv('data.csv')
"""
            except Exception as e:
                logging.error(f"Failed to process plot_data: {e}")
                data_context = f"Note: plot_data provided but could not be processed: {e}"
        
        prompt = f"""
        Based on the following paper plan and research content, suggest 3-5 plots that would enhance this paper.
        
        Paper Plan:
        {plan}
        
        Research Content:
        {content[:2000]}
        
        {data_context}
        
        Task Context:
        {task}
        
        For each plot suggestion, provide the following format:
        
        PLOT 1:
        Description: [Brief description of what the plot shows]
        Rationale: [Why this plot is useful for the paper]
        Code:
        ```python
        [Python matplotlib/seaborn code to generate the plot]
        ```
        
        PLOT 2:
        Description: [Brief description]
        Rationale: [Why useful]
        Code:
        ```python
        [Python code]
        ```
        
        Continue for 3-5 plots total. Generate plots that are relevant to the paper's topic and would provide valuable visual insights.
        """
        
        messages = [
            SystemMessage(content="You are an expert data visualization specialist who creates meaningful plots for academic papers."),
            HumanMessage(content=prompt)
        ]
        
        response = self.model.invoke(messages)
        try:
            import uuid
            import re
            
            response_text = response.content.strip()
            suggested_plots = []

            plot_sections = re.split(r'\bPLOT\s+\d+:', response_text)
            
            for i, section in enumerate(plot_sections[1:], 1):
                desc_match = re.search(r'Description:\s*(.+?)(?=\n\w+:|$)', section, re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else f"Visualization {i}"
                filename_base = re.sub(r'[^\w\s-]', '', description.lower())
                filename_base = re.sub(r'\s+', '_', filename_base)[:30]
                if not filename_base:
                    filename_base = f"plot_{i}"

                rationale_match = re.search(r'Rationale:\s*(.+?)(?=\n\w+:|$)', section, re.DOTALL)
                rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided"

                code_match = re.search(r'```python\s*\n(.*?)\n```', section, re.DOTALL)
                code = code_match.group(1).strip() if code_match else ""
                
                if description or code:  # Only add if we have some content
                    suggested_plots.append({
                        "id": str(uuid.uuid4()),
                        "description": description,
                        "code": code,
                        "rationale": rationale,
                        "approved": False,
                        "filename_base": filename_base
                    })
                
        except Exception as e:
            logging.error(f"Failed to parse plot suggestions: {e}")
            # Fallback: create a simple plot suggestion
            suggested_plots = [{
                "id": str(uuid.uuid4()),
                "description": "Sample visualization",
                "code": "import matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\n\nplt.figure(figsize=(8, 6))\nplt.plot(x, y)\nplt.title('Sample Plot')\nplt.xlabel('X values')\nplt.ylabel('Y values')\nplt.show()",
                "rationale": "A basic visualization to demonstrate the concept",
                "approved": False
            }]
        
        return {
            "state": self.name,
            "suggested_plots": suggested_plots,
            "draft": plan  # Keep the plan as draft for display
        }


class PlotApprovalAgent(State):
    def __init__(self, model: ChatOpenAI):
        super().__init__(model, "plot_approval")

    def run(self, state: AgentState, config: dict) -> dict:
        """
        Handle plot approval/rejection based on user input.
        """
        logging.info(f"state {self.name}: running")
        
        suggested_plots = state.get("suggested_plots", [])
        instruction = config["configurable"].get("instruction", "")
        
        logging.info(f"PlotApprovalAgent received instruction: '{instruction}'")
        
        if instruction:
            if "apply individual selections" in instruction.lower():
                logging.info("Processing checkbox selections - approval status already updated")
            elif "approve" in instruction.lower():
                if "all" in instruction.lower():
                    for plot in suggested_plots:
                        plot["approved"] = True
                    logging.info("Approved all plots")
                else:
                    plot_ids = [s.strip() for s in instruction.lower().replace("approve", "").strip().split(",")]
                    for plot_id in plot_ids:
                        try:
                            idx = int(plot_id) - 1 
                            if 0 <= idx < len(suggested_plots):
                                suggested_plots[idx]["approved"] = True
                                logging.info(f"Approved plot {idx + 1}")
                        except (ValueError, IndexError):
                            pass
            elif "reject" in instruction.lower():
                if "all" in instruction.lower():
                    for plot in suggested_plots:
                        plot["approved"] = False
                    logging.info("Rejected all plots")
                else:
                    plot_ids = [s.strip() for s in instruction.lower().replace("reject", "").strip().split(",")]
                    for plot_id in plot_ids:
                        try:
                            idx = int(plot_id) - 1
                            if 0 <= idx < len(suggested_plots):
                                suggested_plots[idx]["approved"] = False
                                logging.info(f"Rejected plot {idx + 1}")
                        except (ValueError, IndexError):
                            pass
        
        return {
            "state": self.name,
            "suggested_plots": suggested_plots,
            "draft": self._format_plot_summary(suggested_plots)
        }
    
    def _format_plot_summary(self, plots: list) -> str:
        """Format plots for display in the UI."""
        if not plots:
            return "No plots suggested."
        
        summary = "## Suggested Plots\n\n"
        for i, plot in enumerate(plots, 1):
            status = "✅ APPROVED" if plot.get("approved") else "❌ Pending"
            summary += f"### Plot {i}: {plot.get('description', 'Untitled')} [{status}]\n\n"
            summary += f"**Rationale:** {plot.get('rationale', 'No rationale provided')}\n\n"
            summary += f"```python\n{plot.get('code', 'No code provided')[:300]}...\n```\n\n"
        
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
        
        suggested_plots = state.get("suggested_plots", [])
        plot_data = state.get("plot_data", "")
        generated_images = []
        
        # Initialize CodeAPI if available
        codeapi_url = os.environ.get("CODEAPI_URL")
        logging.info(f"CODEAPI_URL: {codeapi_url}")
        if not codeapi_url:
            logging.error("CODEAPI_URL not configured, skipping plot generation")
            logging.error("To enable plot generation, set environment variable: export CODEAPI_URL=http://your-codeapi-server")
            return {
                "state": self.name,
                "generated_plot_images": [],
                "draft": "Plot generation skipped: CODEAPI_URL not configured. Set CODEAPI_URL environment variable to enable plot generation."
            }
        
        try:
            codeapi = CodeAPI(codeapi_url)
            logging.info(f"Successfully initialized CodeAPI with URL: {codeapi_url}")
        except Exception as e:
            logging.error(f"Failed to initialize CodeAPI: {e}")
            return {
                "state": self.name,
                "generated_plot_images": [],
                "draft": f"Plot generation failed: Could not connect to CodeAPI at {codeapi_url}. Error: {e}"
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
            if not plot.get("approved", False):
                continue
                
            plot_code = plot.get("code", "")
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
            
            if plot_data:
                try:
                    import pandas as pd
                    from io import StringIO
                    import os
                    
                    # Check if plot_data is a file path
                    if os.path.isfile(plot_data):
                        # Read the CSV file and pass it to CodeAPI
                        with open(plot_data, 'r', encoding='utf-8') as f:
                            csv_content = f.read()
                        
                        filename = os.path.basename(plot_data)
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
                        logging.info("Prepared inline CSV data as data.csv for CodeAPI upload")
                        
                except Exception as e:
                    logging.error(f"Failed to prepare plot data: {e}")
                    data_bootstrap = "# No data available\n"
            
            full_code = matplotlib_setup + "\n" + data_bootstrap + "\n" + plot_code + "\n" + capture_code
            
            try:
                logging.info(f"Executing plot {i+1}: {plot.get('description', 'Unknown')}")
                logging.debug(f"Plot code length: {len(full_code)} characters")
                if files:
                    logging.info(f"Uploading {len(files)} files to CodeAPI: {list(files.keys())}")
                
                # Execute via CodeAPI with files
                result = codeapi.run_python(full_code, inputs=files)
                logging.info(f"CodeAPI execution result for plot {i+1}: {type(result)}")
                
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
                        filename_base = plot.get("filename_base", f"plot_{i+1}")
                        if len(images_b64) > 1:
                            filename = f"{filename_base}_{j+1}.png"
                        else:
                            filename = f"{filename_base}.png"
                        
                        img_path = images_dir / filename
                        
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        
                        # Store both the full path and the relative path for markdown
                        relative_path = f"images/{filename}"
                        generated_images.append({
                            "path": str(img_path),
                            "relative_path": relative_path,
                            "description": plot.get("description", "Generated plot"),
                            "rationale": plot.get("rationale", "")
                        })
                        logging.info(f"Saved plot image: {img_path}")
                else:
                    logging.warning(f"No images generated for plot {i+1}")
                    
            except Exception as e:
                logging.error(f"Failed to execute plot {i+1}: {e}")
        
        summary = f"Generated {len(generated_images)} plot images from {len([p for p in suggested_plots if p.get('approved')])} approved plots."
        
        return {
            "state": self.name,
            "generated_plot_images": generated_images,
            "draft": summary
        }
