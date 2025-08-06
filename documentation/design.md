## 🚀 Project Title: **LearnMate: Visual Wiki Generator**

### 🎯 Goal

Build a multi-agent system that **generates comprehensive, visually structured wikis** on any topic, with automatically generated diagrams and clear explanations — all accessible via a **Streamlit web interface**.

---

### 🧠 User Persona

A student/professional/enthusiast who:

- Wants a fast but comprehensive grasp of a topic
- Prefers **visually enhanced** content (diagrams, tables, flowcharts)
- Values clear, structured explanations
- Appreciates well-organized information

---

## 🔧 Multi-Agent Architecture

<img src="./multi_agent_design_flowchart.png" width="50%" alt="Multi-Agent Architecture">

### 1. **Research Agent**

- **Input**: Topic from user
- **Functions**:
  - Performs comprehensive web search
  - Extracts key subtopics and concepts
  - Identifies visualization opportunities
  - Gathers reliable source material
- **Output**: Research data with source citations

### 2. **Planner Agent**

- **Input**: Research results
- **Functions**:
  - Creates logical content structure
  - Plans section layout and flow
  - Identifies visual aid placement
  - Distributes content between text and diagrams
- **Output**: Content plan and visual requirements

### 3a. **Content Writer Agent** _(Parallel Path 1)_

- **Input**: Content plan from Planner
- **Functions**:
  - Generates educational content
  - Structures text for clarity
  - Maintains consistent style
  - Integrates examples
- **Output**: Draft content for validation

### 3b. **Mermaid Generator** _(Parallel Path 2)_

- **Input**: Visual requirements from Planner
- **Functions**:
  - Creates Mermaid.js diagrams
  - Designs clear visual layouts
  - Optimizes diagram structure
- **Output**: Draft diagrams for validation

### 4a. **Content Validator**

- **Input**: Draft content
- **Functions**:
  - Verifies accuracy
  - Checks completeness
  - Ensures clarity
  - Provides feedback loop to Content Writer
- **Output**: Validated content or revision requests

### 4b. **Mermaid Validator**

- **Input**: Draft diagrams
- **Functions**:
  - Validates diagram syntax
  - Checks visual clarity
  - Ensures diagram usefulness
  - Provides feedback loop to Mermaid Generator
- **Output**: Validated diagrams or revision requests

### Parallel Processing Note

Paths 3a+4a and 3b+4b run in parallel for efficiency

---

## 🌐 User Interface: **Streamlit Frontend**

### 📌 Layout Overview

| UI Section                 | Function                                                 |
| -------------------------- | -------------------------------------------------------- |
| **Topic Input Box**        | User enters any topic (e.g., “How DNS Works”)            |
| **“Generate Wiki” Button** | Triggers the agent pipeline                              |
| **Generated Wiki Viewer**  | Displays structured output with expandable sections      |
| **Diagrams & Images**      | Renders inline visuals using DALL·E or Mermaid           |
| **Tutor Chat Area**        | Simulated back-and-forth tutor dialog                    |
| **Progress Tracker**       | Visual indicator of user’s learning journey              |
| **Feedback Box**           | User gives optional feedback or asks to revisit sections |

### Streamlit Features Used

- `st.chat_message` for Tutor interaction
- `st.expander` for collapsible content sections
- `st.image` for visual rendering
- `st.session_state` for progress tracking
- (Optional) Audio using `st.audio` if TTS is enabled

---

## ✅ MVP

- Topic Input → Generates a 3–4 section visual wiki
- At least 1 Mermaid diagram per section
- Streamlit UI showing:  
  - Wiki content with expandable sections
  - Visuals with Mermaid diagrams
- No user login/auth or persistence needed

---

## ✨ Optional Enhancements (Stretch Goals)

| Feature                       | Value                                      |
| ---------------------------- | ------------------------------------------ |
| Export Wiki as PDF           | Easy sharing and offline access            |
| Multiple diagram styles      | Support for more visual formats            |
| Custom diagram themes        | Consistent visual styling                  |
| LLM-RAG integration         | Better content grounding via external PDFs |

---

## 🧪 Example Topics to Try

-   "Intro to Kafka & Event Streaming"
-   "How Git Internally Works"
-   "Basics of Prompt Engineering"
-   "Data Warehousing vs Data Lakes"
