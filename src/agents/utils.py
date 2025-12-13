# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from src.utils.models import WorkflowLog


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


if __name__ == "__main__":
    print(is_interactive())
