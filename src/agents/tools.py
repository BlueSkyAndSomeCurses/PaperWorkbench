# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import os

from decouple import config
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from tavily import TavilyClient

from .pubmed import PubMedAPIWrapper

tavily = TavilyClient(config("TAVILY_API_KEY"))

pubmed = PubmedQueryRun()
pubmed.api_wrapper = PubMedAPIWrapper()
arxiv = ArxivQueryRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
python_repl = PythonREPL()
python_repl = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

tools_list = [wikipedia, python_repl, arxiv, pubmed]

tools = {tool.name: tool for tool in tools_list}
