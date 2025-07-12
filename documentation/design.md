## 🚀 Project Title: **LearnMate: Visual Wiki + Personalized Tutor**

### 🎯 Goal:

Build a multi-agent AI system that **generates a visually structured wiki** on a given topic and **tutors the user interactively**, adapting based on their understanding and preferences — all accessible via a **Streamlit web interface**.

---

### 🧠 User Persona:

A student/professional/enthusiast who:

-   Wants a fast but comprehensive grasp of a topic
-   Prefers **visually enhanced** learning (diagrams, tables, flowcharts)
-   Likes interactive guidance and feedback
-   Appreciates a **personalized learning path**

---

## 🔧 Multi-Agent Architecture

![Multi-Agent Architecture](./multi_agent_design_flowchart.png)

### 1. **Research Agent (Content Curator)**

-   Accepts a topic input
-   Performs targeted search or document parsing
-   Identifies key concepts and knowledge gaps
-   Flags sections that would benefit from visuals

### 2. **Structure & Design Agent (Wiki Architect)**

-   Designs a logical content structure (sections, flow)
-   Decides on where to insert visual elements (flowcharts, tables, diagrams)
-   Outputs a wiki "blueprint"

### 3. **Content Generator Agent**

-   Writes well-structured, modular content per blueprint
-   Integrates image/diagram placeholders or generates visuals via:

    -   Mermaid.js (flowcharts)
    -   DALL·E/Stable Diffusion (illustrative images)

### 4. **Personalized Tutor Agent**

-   Walks the user through the content in **small chunks**
-   Asks understanding-check questions after each chunk
-   Adapts:

    -   If user answers correctly → proceed or offer deep dive
    -   If incorrect/unsure → explain again with a simpler example or analogy

### 5. **Reviewer Agent** _(Optional but useful for auto-feedback)_

-   Runs content through checks for:

    -   Accuracy
    -   Clarity
    -   Completion

-   Suggests improvements before Tutor begins

---

## 🌐 User Interface: **Streamlit Frontend**

### 📌 Layout Overview:

| UI Section                 | Function                                                 |
| -------------------------- | -------------------------------------------------------- |
| **Topic Input Box**        | User enters any topic (e.g., “How DNS Works”)            |
| **“Generate Wiki” Button** | Triggers the agent pipeline                              |
| **Generated Wiki Viewer**  | Displays structured output with expandable sections      |
| **Diagrams & Images**      | Renders inline visuals using DALL·E or Mermaid           |
| **Tutor Chat Area**        | Simulated back-and-forth tutor dialog                    |
| **Progress Tracker**       | Visual indicator of user’s learning journey              |
| **Feedback Box**           | User gives optional feedback or asks to revisit sections |

### Streamlit Features Used:

-   `st.chat_message` for Tutor interaction
-   `st.expander` for collapsible content sections
-   `st.image` for visual rendering
-   `st.session_state` for progress tracking
-   (Optional) Audio using `st.audio` if TTS is enabled

---

## ✅ MVP (Target in 2 Days)

-   Topic Input → Generates a 3–4 section visual wiki
-   At least 1 diagram + 1 illustrative image
-   Streamlit UI showing:

    -   Wiki content
    -   Visuals
    -   Tutor interaction (basic Q\&A)

-   No user login/auth or persistence needed

---

## ✨ Optional Enhancements (Stretch Goals)

| Feature                            | Value                                      |
| ---------------------------------- | ------------------------------------------ |
| TTS Integration (e.g., ElevenLabs) | Auditory learners                          |
| Quiz Mode                          | Post-learning assessment                   |
| Bookmark / Export Wiki as PDF      | Revisit or share                           |
| User Profile & Progress Save       | Personalization over time                  |
| LLM-RAG integration                | Better content grounding via external PDFs |

---

## 🧪 Example Topics to Try

-   "Intro to Kafka & Event Streaming"
-   "How Git Internally Works"
-   "Basics of Prompt Engineering"
-   "Data Warehousing vs Data Lakes"
