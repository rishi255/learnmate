# Streamlit UI Design Document

## Overview

LearnMate's frontend is implemented as a Streamlit web application that provides an intuitive interface for generating and viewing visual wikis. The UI focuses on simplicity and effectiveness while maintaining a smooth user experience.

## Interface Layout

### Current Implementation

#### Main Sections

| UI Section | Function |
| ---------- | -------- |
| **Sidebar** | Contains app title and state file upload |
| **Topic Input Box** | User enters any topic (e.g., "How DNS Works") |
| **Generate Wiki Button** | Triggers the wiki generation pipeline |
| **Wiki Content View** | Renders markdown content with Mermaid diagrams |
| **Error Display** | Shows detailed error messages with tracebacks |

### Implemented Features

| UI Section | Function |
| ---------- | -------- |
| **Generated Wiki Viewer** | Displays structured markdown with automatic section organization |
| **Diagrams & Visuals** | Native Mermaid diagram rendering through Streamlit markdown |

### Planned Extensions

| UI Section | Function |
| ---------- | -------- |
| **Interactive Features** | Enhanced section navigation and content interaction |
| **Feedback Box** | User feedback collection and wiki editing capabilities |
| **DALL·E Integration** | AI-generated images for supported sections |
| **Audio Controls** | Optional TTS playback (future enhancement) |

## Key Features

### Core Components

1. **Project Structure**

```
frontend/
├── app.py              # Main Streamlit application
├── components/         # Core UI components
│   ├── wiki_viewer.py  # Wiki content display
│   └── sidebar.py      # Sidebar components
└── utils/             # UI utility functions
    └── state.py       # State management
```

1. **Session State**

```python
st.session_state = {
    'current_topic': str,     # Current topic being processed
    'error_message': str,     # Current error message if any
    'wiki_path': str,        # Path to the generated wiki file
}
```

1. **Component Design**

#### Main App (`app.py`)

- Initializes session state
- Handles topic input and generation flow
- Manages state file upload workflow
- Provides error handling and user feedback
- Coordinates between components

#### Wiki Viewer (`wiki_viewer.py`)

- Reads and displays markdown content
- Handles Mermaid diagram rendering
- Simple, file-based content display

#### Sidebar (`sidebar.py`)

- Shows application title
- Manages state file upload
- Validates uploaded JSON files

#### State Management (`state.py`)

- Minimal session state initialization
- Error message handling
- Clean state management

### Current Features

1. **Core Functionality**
   - Topic input with validation
   - Wiki generation with progress feedback
   - State file upload for resumption
   - Direct markdown file rendering

2. **User Interface**
   - Clean, minimal design
   - Clear error messages with tracebacks
   - Loading states and progress indicators
   - Native Mermaid diagram support

3. **State Management**
   - File-based state preservation
   - Simple session state
   - Efficient error handling

### Future Enhancements (Planned)

1. **Phase 2: UI Improvements**
   - Collapsible table of contents
   - Better progress tracking
   - Enhanced error recovery
   - Expandable sections

2. **Phase 3: Advanced Features**
   - URL/document context support
   - Interactive Mermaid diagrams
   - Export capabilities (PDF)
   - Responsive design
   - Performance optimizations

### Design Principles

- Uses native Streamlit features where possible
- Prioritizes simplicity and reliability
- File-based approach for efficiency
- Clear error handling and user feedback
- No complex state management needed

## Implementation Phases

### Phase 1: Basic UI (Current Focus)

1. Topic input interface
   - Text input field
   - Generate button
   - State file upload
2. Basic wiki display
   - Collapsible sections
   - Table of contents
   - Basic Mermaid rendering
3. Core state management
   - State file handling
   - Session state setup
4. Error message display
   - Basic error notifications
   - Clear error states

### Phase 2: Progress Tracking (Next Iteration)

1. Agent status indicators
2. Overall progress tracking
3. Enhanced error handling

### Phase 3: Enhanced Features (Future)

1. Export capabilities
2. Interactive editing
3. Advanced visualizations
4. Performance optimizations

## UI Layout

### Main Page

```
├── Sidebar
│   ├── Application Title
│   └── State File Upload (JSON)
├── Main Content
│   ├── Topic Input
│   │   ├── Text Input
│   │   └── Generate Button
│   ├── Progress Indicators
│   │   ├── Loading Spinners
│   │   └── Success/Info Messages
│   ├── Wiki Display
│   │   ├── Markdown Content
│   │   └── Mermaid Diagrams
│   └── Error Display
       ├── Error Messages
       └── Full Traceback (if any)
├── Temporary Storage
│   └── temp/
       └── Uploaded State Files
```

## Notes

- Using default Streamlit theming for modern look
- Single page design for simplicity
- Read-only content during generation
- No authentication required
- Native Streamlit features preferred over custom implementations
- Error handling limited to clear message display
