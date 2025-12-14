# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from src.utils.models import RelevantFile, WorkflowLog
from utils.file_handlers import handle_file_reading_for_model


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


def create_log_entry(message: AnyMessage, state_name: str) -> WorkflowLog:
    side = "Unknown"
    if isinstance(message, AIMessage):
        side = "AI"
    elif isinstance(message, HumanMessage):
        side = "User"
    else:
        side = "System"

    return WorkflowLog(step=state_name, side=side, message=str(message.content))


def create_document_prompt(rel_file: RelevantFile) -> str:
    return f"""
                Use this content:\n,
            {handle_file_reading_for_model(rel_file.file_path)}
                It's concise description {rel_file.description}
            """ + "\n".join(
        [
            f"Use it in section {rel_file_application.stage_name}, this document should be used in this section to {rel_file_application.application_desc}"
            for rel_file_application in rel_file.application
        ]
    )
