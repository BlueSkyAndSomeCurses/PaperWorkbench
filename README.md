# Introduction

It is a multi-agent system that helps you organize and write documents.  

![AdvisorPicture](https://github.com/user-attachments/assets/dbbed542-4d24-4af2-bf83-3d6fc5113c4f)
(c) PhDCommics (www.phdcommics.com) of the advisor and the student

# Before You Run

To run Kiroku, you need an OPENAI_API_KEY and a TAVILY_API_KEY.

To get an OPENAI_API_KEY, you can check https://platform.openai.com/docs/quickstart .

To get a TAVILY_API_KEY, you can check the site https://app.tavily.com/sign-in, and click "Sign in".

You may want to use a tool like `direnv` to manage the environment variables `OPENAI_API_KEY` and `TAVILI_API_KEY` on a per-directory basis. 
This will help you automatically load these variables when you are working within the Kiroku project directory. 
`direnv` supports Linux, macOS, and Windows through WSL.

# Installation

Kiroku supports Python between versions 3.7 and 3.11.

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
- Introduction
- Related Work
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
this time, so you will not see them. I usually put `\n\n` after each instruction to assist the underlying LLM.
- `results`: I usually put it here as I fill this later on.
- `references` are references you want Kiroku to use during its search phase for information.
- `number_of_queries` tells Kiroku how many questions it will generate to Tavily to search for information.
- `max_revisions` tells Kiroku how many times it performs reflection and document writing upon analyzing reflection results
(please note that setting this document to `1`, it means no revision).
- `temperature` is the temperature of the LLM (usually I set it to a small number).

The final YAML is given below:

```yaml
title: "Pythagorean Theorem Proof"
suggest_title: False
generate_citations: False
model_name: "gpt-5-nano"
type_of_document: "Research conference paper"
area_of_paper: "Mathematics"
section_names:
    Introduction: |
        For the following instructions, you should use your own words.
        The section 'Introduction', you should focus on:
        In the first paragraph, describe the problem of finding hypotenus in a triangle.
        Write the main formula of pythagorean triangle. Insert the picture `multi-agent.jpeg`.
    Derivation: |
        Using the derivation described in relevant documents, structure it in a logical manner and write it in paper.

number_of_paragraphs:
    "Introduction": 1
    "Derivation": 1
hypothesis: "We want to prove the pythagorean theorem using squares and triangles."

number_of_queries: 0
max_revisions: 1
temperature: 1

files_descriptions: # file is considered only if its name is mention here
    - file_name: "multi-agent.jpeg"
      description: "An image showing a right triangle with squares on each side to illustrate the Pythagorean theorem. To be used in introduction section"
    - file_name: "train.csv"
      description: "A CSV file containing training data for a machine learning model. To be used in for plots in introduction section" # summary of the document, once provided model won't provide any
    - file_name: "pythagorean_triangles.csv"
      # However, model will summarize for itself and write a list of sections where the document is relevant, together with how to use it for the specific section.
    - file_name: "example.html"
    # - file_name: "example.tex"
    #   description: "This file contains the derivation of the main formula"
    # - file_name: "example_plan.md"
    # - file_name: "some_info.yml"

working_dir: "/Users/vitya/Documents/PaperWorkbench/proj/" # Directory to store all the drafts
output_dir: "/Users/vitya/Documents/PaperWorkbench/proj/example_result/" # Directory to store the final output

latex_template: "IEEE" # LaTeX template to use for formatting the final document. Options: "IEEE", "ACM", "Springer", "Elsevier", "Nature", "Custom"
# You can either provide a CSV file path:
```

There is a script `check_yaml` that checks if the YAML file is consistent and it will not crash Kiroku.

I recommend putting all YAML files right now in the `kikoku/proj` directory. All images should be in `kiroku/proj/images`. 

Because of a limitation of Gradio, you need to specify images as `'/file=images/<your-image-file>'` such as in the example `/file=images/multi-agent.jpeg`.

# Running

I recommend running writer as:

```shell
python -m src.kiroku_app
```

Go to your preferred browser and open `localhost:7860`.

As for instructions, you can try `I liked title 2` or `I liked the original title`.

Whenever you give an instructions you really liked, remember to add it to the `instructions` field.

# License

Apache License 2.0 (see LICENSE.txt)

# References

<p id=1> 1. https://www.youtube.com/watch?v=om7VpIK90vE</p>

<p id=2> 2. Harrison Chase, Rotem Weiss. AI Agents in LangGraph. https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph</p>

# Authors

Claudionor N. Coelho Jr (https://www.linkedin.com/in/claudionor-coelho-jr-b156b01/)

Fabricio Ceolin (https://br.linkedin.com/in/fabceolin)

Luiza N. Coelho (https://www.linkedin.com/in/luiza-coelho-08499112a/)
(looking for a summer internship for summer of 2025 in Business Development, Economics, Marketing)



