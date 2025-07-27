# langchain-tableau v0.4.39.1

<!-- [![PyPI version](https://badge.fury.io/py/langchain-tableau.svg)](https://badge.fury.io/py/langchain-tableau)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Tab-SE/tableau_langchain) -->

This package is mod

This package provides Langchain integrations for Tableau, enabling you to build Agentic tools using Tableau's capabilities within the [Langchain](https://www.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/) frameworks.

Use these tools to bridge the gap between your organization's Tableau data assets and the natural language queries of your users, empowering them to get answers directly from data through conversational AI agents.

![Tableau Logo](https://raw.githubusercontent.com/Tab-SE/tableau_langchain/main/experimental/notebooks/assets/tableau_logo_text.png)

## Installation

```bash
pip install langchain-tableau
```

## Quick Start
Here's a basic example of using the `simple_datasource_qa` tool to query a Tableau Published Datasource with a Langgraph agent:

Define the environment `.env`:
```.env
OPENAI_COMPATIBLE_BASE_URL=
OPENAI_COMPATIBLE_MODEL=

TABLEAU_DOMAIN=
TABLEAU_SITE=
TABLEAU_JWT_CLIENT_ID=
TABLEAU_JWT_SECRET_ID=
TABLEAU_JWT_SECRET=
TABLEAU_API_VERSION=
TABLEAU_USER=
DATASOURCE_LUID=
```

Python sample:
```python
# --- Core Langchain/LangGraph Imports ---
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_tableau.tools.simple_datasource_qa import initialize_simple_datasource_qa

import os
from dotenv import load_dotenv

# 1. Import the parameters
load_dotenv(override=True)

TABLEAU_SERVER = os.getenv('TABLEAU_DOMAIN')
TABLEAU_SITE = os.getenv('TABLEAU_SITE')
TABLEAU_JWT_CLIENT_ID = os.getenv('TABLEAU_JWT_CLIENT_ID')
TABLEAU_JWT_SECRET_ID = os.getenv('TABLEAU_JWT_SECRET_ID')
TABLEAU_JWT_SECRET = os.getenv('TABLEAU_JWT_SECRET')
TABLEAU_API_VERSION = os.getenv('TABLEAU_API_VERSION')
TABLEAU_USER = os.getenv('TABLEAU_USER')
DATASOURCE_LUID = os.getenv('DATASOURCE_LUID')

OPENAI_COMPATIBLE_BASE_URL=os.getenv('OPENAI_COMPATIBLE_BASE_URL')
OPENAI_COMPATIBLE_MODEL=os.getenv('OPENAI_COMPATIBLE_MODEL')

# 2. Initialize your preferred LLM
llm = ChatOpenAI(
    base_url=OPENAI_COMPATIBLE_BASE_URL,
    model=OPENAI_COMPATIBLE_MODEL,
    api_key="dummy",
    temperature=0
)

# 3. Initialize the Tableau Datasource Query tool
analyze_datasource = initialize_simple_datasource_qa(
    domain=TABLEAU_SERVER,
    site=TABLEAU_SITE,
    jwt_client_id=TABLEAU_JWT_CLIENT_ID,
    jwt_secret_id=TABLEAU_JWT_SECRET_ID,
    jwt_secret=TABLEAU_JWT_SECRET,
    tableau_api_version=TABLEAU_API_VERSION,
    tableau_user=TABLEAU_USER,
    datasource_luid=DATASOURCE_LUID,
    tooling_llm_model=OPENAI_COMPATIBLE_BASE_URL ## Not the model just be string the source will call it!
)

# 4. Create a list of tools for your agent
tools = [ analyze_datasource ]

# 5. Define system prompt (optional)
identity = """<Define your identity of AI>"""
system_prompt = f"""{identity} and define the instruction to use tool name: {analyze_datasource.name}"""

# 6. Build the Agent
# This example uses a prebuilt ReAct agent from LangGraph
tableau_agent = create_react_agent(
    model=llm, 
    tools=tools, 
    prompt=system_prompt
)

# 7. Define view for invoke message
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# 8. Run the Agent with a question
question = 'Which states sell the most? Are those the same states with the most profits?'
messages = {"messages": [("user", your_prompt)]}
history = print_stream(tableau_agent.stream(messages, stream_mode="values"))

# Process and display the agent's response
print(history)

```

## Prompt
This `langchain-tableau` package is a sample to demonstrate proof-of-concept LLM integration. The prompt logic (`pkg/langchain_tableau/tools/prompts.py`) is currently hard-coded and will need to be modified to fit your specific data environment.
