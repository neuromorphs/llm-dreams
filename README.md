# llm-dreams
Repository for the LLM-Dreaming project at the 2024 Telluride Neuromorphic Workshop

Introduction

Provide a more detailed description of the project, its purpose, and its goals.
Prerequisites

Before you begin, ensure you have met the following requirements:

    You have installed Git.
    You have installed Miniconda or Anaconda.

Installation

Follow these steps to set up the project on your local machine.
Creating a Conda Environment

    Open a terminal.

    Create a new Conda environment:

    bash

conda create --name myenv python=3.12

Activate the Conda environment:

bash

    conda activate myenv

Installing Ollama

    Install Ollama via Conda:

    bash

    conda install -c ollama ollama

Pulling PHI3

    Use Ollama to pull the PHI3
    package:

    bash

    ollama pull phi3:mini
    ollama run phi3:mini

Test your ollama installation by having a conversation with phi!

Installing LangGraph

    Install LangGraph:

    bash

    pip install langgraph

Usage

The provided `ToT_base.py` is a simple implementation of tree of thoughts, running locally it takes quite a long time to get to an answer but I encourage you to play with the main and subprompts to tailor it to a problem. The default script attempts to solve a new york times word association/grouping problem.

`python ToT_base.py`