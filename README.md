# Introduction

It is a multi-agent system that helps you organize and write documents.  

![AdvisorPicture](https://github.com/user-attachments/assets/dbbed542-4d24-4af2-bf83-3d6fc5113c4f)
(c) PhDCommics (www.phdcommics.com) of the advisor and the student

# Before You Run

To run Kiroku, you need an OPENAI_API_KEY and a TAVILY_API_KEY.

To get an OPENAI_API_KEY, you can check https://platform.openai.com/docs/quickstart .

To get a TAVILY_API_KEY, you can check the site https://app.tavily.com/sign-in, and click "Sign in".

Create and .env file in the repository source containing those key
```
OPENAI_API_KEY=<your openai api key>
TAVILY_API_KEY=<your tavily api key>
```

[CodeAPI](https://codapi.org/) was used for remote code execution. In case you own a paid plan or host server yourself add the url environment file
```
CODEAPI_URL=<your codeapi url>
```

# Installation

### 1. Set up a virtual environment
You can use Python’s `venv` module to create an isolated environment for dependencies. This ensures a clean environment and avoids conflicts with system packages.

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Installation of PANDOC

You need to install PANDOC. As pointed out by Charles Ferreira Gonçalves, in macos, you can do it by executing the following command.

```shell
brew instal pandoc
```

# The Initial Configuration

The initial configuration is specified in an YAML file with the following fields:

- `title` is a suggestion for the title or the final title to use (if `suggest_title` is false).
- `suggest_title` turns on recommendation for titles based on your original title.
- `generate_citations`: if true, it will generate citations and references.
- `type_of_document`: helps the Kiroku define whether it should use more technical terms, or whether we are trying to write children's stories.
- `area_of_paper`: together with `hypothesis`, it helps Kiroku to understand what to write.
- `section_names`: list of sections, as in the example below:
```markdown
section_names:
- Introduction: Instructions to be followed when writing this section
- Related Work: ...
- Architecture of Kiroku
- Results
- Conclusions
- References
```
- `number_of_paragraphs`: instructs Kiroku to write that many paragraphs per section.
```markdown
number_of_paragraphs:
  "Introduction": 4
  "Related Work": 7
  "Architecture of Kiroku": 4
  "Results": 4
  "Conclusions": 3
  -"References": 0
```
- `hypothesis` tells Kiroku whether you want to establish something to be good or bad, and it will define the message.
- `instructions`: as you interact with the document giving instructions like "First paragraph of Introduction should
discuss the revolution that was created with the lauch of ChaGPT", you may want to add some of them to the instruction so that
in the next iteration, Kiroku will observe your recommendations. In Kiroku, `instructions` are appended into the `hypothesis` at
this time, so you will not see them.  We usually put `\n\n` after each instruction to assist the underlying LLM.
- `references` are references you want Kiroku to use during its search phase for information.
- `number_of_queries` tells Kiroku how many questions it will generate to Tavily to search for information.
- `max_revisions` tells Kiroku how many times it performs reflection and document writing upon analyzing reflection results
(please note that setting this document to `1`, it means no revision).
- `temperature` is the temperature of the LLM (usually I set it to a small number).
- `working_dir` is the directory with all the relevant files that agent may need throughout writing paper.
- `output_dir` directory where all the files will be saved
- `files_descriptions` are names of files located in the `working_dir` together with their optional description
- `latex_template` latex template of the available templates of the document. Templates should be installed under `latex_templates/` directory.

There is a script `check_yaml` that checks if the YAML file is consistent and it will not crash Kiroku.

# Running

To run the system simply use: 
```shell
python -m src.kiroku_app
```

Go to your preferred browser and open `localhost:7860`.

As for instructions, you can try `I liked title 2` or `I liked the original title`.

Whenever you give an instructions you really liked, remember to add it to the `instructions` field.

# Implementation details
The workflow is sequential with human-in-the-loop for main part of paper writing process and for plot generation.

After user have uploaded `yaml` config of the future paper, system starts analyzing all the files located in the `working_dir`. Then it generates title ideas (if requested by user), searches for future references in the internet, writes plan for each section. After the plan have been generated it writes the main paper body using all the files and references obtained so far. In parallel, user may generate plots and diagrams using provided tabular data.

When writing the paper user may give additional instructions which will be considered by agent to refine the resulting document. After these steps system finilizes generation process by creating references to used data and converting to LaTeX publication ready style specified by user.

# License

Apache License 2.0 (see LICENSE.txt)

# References

<p id=1> 1. https://www.youtube.com/watch?v=om7VpIK90vE</p>

<p id=2> 2. Harrison Chase, Rotem Weiss. AI Agents in LangGraph. https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph</p>

<p id=3> 3. S. Schmidgall et al., "Agent Laboratory: Using LLM Agents as Research Assistants," arXiv preprint arXiv:2501.04227, Jun. 2025.

<p id=4> 4. Chengwei Liu et al., “A Vision for Auto Research with LLM Agents”, arXiv preprint arXiv:2504.18765, Apr. 2025. 

<p id=5> 5. Q. Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation," presented at the LLM Agents Workshop at ICLR 2024, Vienna, Austria, May 2024.

<p id=6> 6. K. Goswami, "PlotGen: Multi-Agent LLM-based Scientific Data Visualization via Multimodal Feedback," arXiv preprint arXiv:2502.00988, Feb. 2025.

<p id=7> 7. A. Plaat, M. van Duijn, N. van Stein, M. Preuss, P. van der Putten, and K. J. Batenburg, "Agentic Large Language Models, a survey," arXiv preprint arXiv:2503.23037, Mar. 2025.

<p id=8> 8. B. Georgiev, J. Gómez-Serrano, T. Tao, and A. Z. Wagner, "MATHEMATICAL EXPLORATION AND DISCOVERY AT SCALE," arXiv preprint arXiv:2511.02864, Nov. 2025.

# Authors

[Claudionor N. Coelho Jr](https://www.linkedin.com/in/claudionor-coelho-jr-b156b01/)

[Fabricio Ceolin](https://br.linkedin.com/in/fabceolin)

[Luiza N. Coelho](https://www.linkedin.com/in/luiza-coelho-08499112a/)
(looking for a summer internship for summer of 2025 in Business Development, Economics, Marketing)

[Anton Valihurskyi](https://www.linkedin.com/in/anton-valihurskyi-1b6a64347/)

[Mariia Onyschuk](https://www.linkedin.com/in/mariia-onyshchuk-230a5633b/)

[Maksym-Vasyl Tarnavskyi](https://www.linkedin.com/in/maksym-vasyl-tarnavskyi-510366352/)



