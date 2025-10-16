from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, Field


class RelevantaFileApplication(BaseModel):
    stage_name: str = ""
    application_desc: str = ""


class RelevantFile(BaseModel):
    file_path: Path = Path()
    description: str = ""
    application: list[RelevantaFileApplication] = []


class PaperConfig(BaseModel):
    # === From YAML ===
    title: str = ""
    suggest_title: bool = False
    generate_citations: bool = False
    model_name: str = "gpt-5-nano"
    type_of_document: str = ""
    area_of_paper: str = ""
    section_names: dict = {}
    number_of_paragraphs: dict[str, int] = {}
    hypothesis: str = ""
    instructions: str = ""
    results: str = ""
    references: list = []
    number_of_queries: int = 0
    max_revisions: int = 1
    temperature: float = 0.0

    working_dir: Path = Path()
    output_dir: Path = Path()

    # === From your original model ===
    sentences_per_paragraph: int = 4
    state: str = "suggest_title"
    draft: str = ""
    revision_number: int = 1
    messages: list = []
    review_instructions: list = []
    review_topic_sentences: list = []

    relevant_files: list[RelevantFile] = []

    latex_template: str = "IEEE"


class AgentState(PaperConfig):
    task: str = ""
    plan: str = ""
    critique: str = ""
    cache: set = set()
    content: list = []
    latex_draft: str = ""
