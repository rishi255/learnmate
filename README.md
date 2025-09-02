# 🤖 LearnMate: AI-Powered Visual Wiki Generator

Transform any topic into a comprehensive wiki in minutes! LearnMate uses a team of specialized AI agents to research, structure, and visualize complex topics with stunning diagrams and clear explanations.

## ✨ Why LearnMate?

- 🎯 **One-Click Wiki Creation** - Just provide a topic, let AI do the rest
- 🧠 **Smart Two-Phase Research** - Deep exploration ensures comprehensive coverage
- 🎨 **Rich Visualizations** - Auto-generated diagrams, tables, and images
- 📝 **Intelligent Organization** - Perfect structure and flow for easy understanding
- ⚡ **Parallel Processing** - Lightning-fast generation with multiple AI agents
- 💾 **Never Lose Progress** - Built-in state management for interrupted runs

## 🚀 Quick Start

> Transform any topic into a visual wiki in 3 simple steps!

1. 📦 Set up dependencies (choose one):

   ```bash
   # Option 1: Using uv sync (Recommended)
   pip install uv # if uv is not installed
   uv venv
   uv sync

   # Option 2: Using uv pip
   pip install uv # if uv is not installed
   uv venv
   uv pip install -r requirements.txt

   # Option 3: Using pip
   python -m venv .venv
   pip install -r requirements.txt
   ```

2. 🎬 Start the generator:

   ```bash
   python backend/main.py
   ```

3. 💡 Enter your topic and watch the magic happen!

4. 🔄 Resume an interrupted run (optional):

   ```bash
   # State is automatically saved after each successful node
   python backend/main.py --state outputs/Your_Topic/saved_wiki_state.json
   ```

   This will continue from the last successful node, skipping completed steps.

   You can also manually edit the state file to modify or retry specific steps.

   > [!WARNING]
   > When editing the state file, ensure you maintain the sequential dependency of steps - a state key that appears later in the JSON (e.g., `wiki_content`) requires all previous state keys (e.g., `initial_research`, `deep_research`, `structure_plan`) to be present and valid.

## 📚 Documentation

Want to know more? Explore our detailed documentation:

- 📖 [Design Documentation](documentation/design.md) - Dive deep into how it works
- 🔍 [Architecture Diagram](documentation/multi_agent_design_flowchart.png) - See the system design

### Architecture Overview

<img src="documentation/multi_agent_design_flowchart.png" alt="Multi-Agent Architecture" width="300"/>

## 💾 Smart State Management

Never lose your progress! The system automatically saves your work:

- ✅ Automatic checkpoints after each successful step
- 📁 Organized state files in `outputs/[Topic_Name]/`
- 🔄 Easy resumption from any interruption

## 📊 Project Status

What's ready and what's cooking:

### Available Now ✨

- 🎯 Smart Research & Planning
- 📝 Dynamic Content Generation
- 🎨 Visual Generation
- 💾 State Management

### On the Horizon 🌅

- 🔍 Advanced Validation Layer
- 🌐 Web Interface with Streamlit
