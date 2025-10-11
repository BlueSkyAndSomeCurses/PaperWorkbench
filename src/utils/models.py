from pydantic import BaseModel, Field


class PaperConfig(BaseModel):
    # === From YAML ===
    title: str = ""
    suggest_title: bool = False
    generate_citations: bool = False
    model_name: str = "gpt-5-nano"
    type_of_document: str = ""
    area_of_paper: str = ""
    section_names: list[str] = []
    number_of_paragraphs: dict[str, int] = {}
    hypothesis: str = ""
    instructions: str = ""
    results: str = ""
    references: list[str] = []
    number_of_queries: int = 0
    max_revisions: int = 1
    temperature: float = 0.0

    # === From your original model ===
    sentences_per_paragraph: int = 4
    state: str = "suggest_title"
    draft: str = ""
    revision_number: int = 1
    messages: list[str] = []
    review_instructions: list[str] = []
    review_topic_sentences: list[str] = []
