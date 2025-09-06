# 🤖 LearnMate: AI-Powered Visual Wiki Generator

Transform any topic into a comprehensive wiki in minutes! LearnMate uses a team of specialized AI agents to research, structure, and visualize complex topics with stunning diagrams and clear explanations.

## ✨ Why LearnMate?

- 🎯 **One-Click Wiki Creation** - Just provide a topic, let AI do the rest
- 🧠 **Smart Two-Phase Research** - Deep exploration ensures comprehensive coverage
- 🎨 **Rich Visualizations** - Auto-generated diagrams, tables, and images
- 📝 **Intelligent Organization** - Perfect structure and flow for easy understanding
- ⚡ **Parallel Processing** - Lightning-fast generation with multiple AI agents
- 💾 **Never Lose Progress** - Built-in state management for interrupted runs

## 🚀 Quick Start (Frontend UI)

1. 📦 Set up dependencies (choose one):

   ```bash
   # Recommended: using uv
   pip install uv  # if uv is not installed
   uv venv
   source .venv/bin/activate   # MacOS/Linux
   # or
   .venv\Scripts\activate   # Windows
   uv sync

   # Alternative: using pip
   python -m venv .venv
   source .venv/bin/activate   # MacOS/Linux
   # or
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. 🎬 Run the Streamlit UI:

   ```bash
   streamlit run frontend/app.py
   ```

   This launches a browser interface where you can enter a topic, generate a wiki, and manage saved states.

3. 🔄 Resume from a saved state (optional)

   The pipeline saves state after each successful node. If a run is interrupted, you can resume from the last checkpoint by uploading the saved state file from the sidebar.

   The saved state file will be placed at `outputs/Your_Topic/saved_wiki_state.json`.

   You can also manually edit the state file to modify or retry specific steps.

> [!WARNING]
> When manually editing the state file, always maintain sequential dependency in the order the keys appear (e.g., `wiki_content` requires `structure_plan` which requires `combined_research`).

## 🛠️ CLI Usage (Optional)

If you prefer to use the CLI directly:

1. 🎬 Run the generator:

   ```bash
   python backend/main.py
   ```

2. 🔄 Resume from a saved state (optional)  

   The pipeline saves state after each successful node. If a run is interrupted, you can resume from the last checkpoint using:

   ```bash
   python backend/main.py --state outputs/Your_Topic/saved_wiki_state.json
   ```

   You can also manually edit the state file to modify or retry specific steps.

> [!WARNING]
> When manually editing the state file, always maintain sequential dependency in the order the keys appear (e.g., `wiki_content` requires `structure_plan` which requires `combined_research`).

## ⚙️ Configuration

The wiki generation process can be configured via the following files:

### config/config.yaml

- `research_model_llm`: LLM for the initial research phase (e.g., gpt-4o-mini)
- `planner_model_llm`: LLM for planning content structure
- `content_writer_model_llm`: LLM for generating text content
- `design_coder_model_llm`: LLM for generating visual code (Mermaid/table/image specs)
- `mermaid_api_base_url`: Base URL for the Mermaid render service
  - Default: [https://mermaid.ink](https://mermaid.ink)
  - To use a local renderer: set to [http://localhost:3000](http://localhost:3000) (or your local service URL)
- `reasoning_strategies`: Named strategies that prompts can reference (e.g., CoT, ReAct, Self-Ask)

### .streamlit/config.toml (Theme)

Streamlit loads this file automatically to style the UI.

Keys:

- primaryColor: Primary accent color
- backgroundColor: Main background color
- secondaryBackgroundColor: Sidebar/secondary panel background
- textColor: Default text color
- font: One of "sans serif", "serif", or "monospace"

Example:

```toml
[theme]
primaryColor = "#5591f5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Environment variables (.env or system)

Set at least one provider key and select the corresponding model in config/config.yaml:

- `GROQ_API_KEY`
- `OPENAI_API_KEY`

## 📚 Design Docs and Architecture

- [Multi-agent Architecture design](documentation/backend_design.md)
- [Streamlit UI design](documentation/streamlit_ui_design.md)

### Architecture Overview

<img src="documentation/multi_agent_design_flowchart.png" alt="Multi-agent Architecture" width="300"/>

## 🧩 Mermaid Rendering

- Mermaid diagrams are rendered automatically in the final wiki output.
- The render service can be optionally configured via `config/config.yaml` (key: `mermaid_api_base_url`).
- If rendering fails (e.g., service unavailable), the original \`\`\`\`mermaid\` fenced code block is embedded as a fallback.

## 💾 Smart State Management

Never lose your progress! The system automatically saves your work:

- ✅ Automatic checkpoints after each successful step
- 📁 Organized state files in `outputs/[Topic_Name]/`
- 🔄 Easy resumption from any interruption

## 📊 Project Status

Available now:

- Smart research & planning
- Dynamic content generation
- Visual generation
- Clean UI
- State management
- Enhanced tool calling

On the Horizon:

- Advanced validation layer
- Additional UI enhancements in Streamlit
