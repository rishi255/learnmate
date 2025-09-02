# Streamlit UI Design Document

## Scope Overview

### In Scope (Phase 1)

1. **Topic Input and Generation**
   - Clean text input field
   - "Generate Wiki" button with loading state
   - State file upload option for resuming generation

2. **Wiki Display**
   - Collapsible table of contents
   - Expandable sections for content
   - Syntax-highlighted code blocks
   - Basic Mermaid diagram support (native Streamlit rendering)

3. **State Management**
   - Leveraging existing state management system
   - State file upload/download functionality
   - Session state persistence

### Future Enhancements (Out of Scope)

1. URL/document context support
2. Real-time progress tracking
3. Interactive Mermaid diagrams (beyond native support)
4. Responsive tables and image zoom
5. Interactive content editing
6. Export options (PDF, etc.)
7. Performance optimizations
8. Advanced error handling and recovery
9. Responsive design
10. Authentication system

## Technical Design

### 1. File Structure

```
streamlit/
├── app.py              # Main Streamlit application
├── components/         # Core UI components
│   ├── wiki_viewer.py  # Wiki content display
│   └── sidebar.py      # Sidebar components
└── utils/             # UI utility functions
    └── state.py       # State management
```

### 2. State Management

#### Session State Structure

```python
st.session_state = {
    'current_topic': str,          # Current topic being processed
    'wiki_state': dict,            # Main state dictionary from core system
    'expanded_sections': set,      # Track expanded/collapsed sections
    'error_message': str          # Current error message if any
}
```

### 3. Agent Integration System

#### Event Emission Architecture

The integration between the core system and Streamlit UI will use an event-driven approach:

```python
from typing import Protocol, Dict, Any
from enum import Enum

class AgentEvent(Enum):
    STARTED = "started"
    COMPLETED = "completed"
    ERROR = "error"

class AgentEventEmitter(Protocol):
    def emit(self, agent_name: str, event_type: AgentEvent, data: Dict[str, Any] = None) -> None:
        """Emit an agent event"""
        pass

class StreamlitEventHandler:
    def __init__(self):
        self._callbacks: Dict[AgentEvent, callable] = {}
    
    def handle_event(self, agent_name: str, event_type: AgentEvent, data: Dict[str, Any] = None):
        """Handle events from agents and update Streamlit UI accordingly"""
        if event_type == AgentEvent.STARTED:
            st.session_state[f"{agent_name}_status"] = "Running"
        elif event_type == AgentEvent.COMPLETED:
            st.session_state[f"{agent_name}_status"] = "Complete"
            st.session_state.wiki_state = data['state']  # Update wiki state
        elif event_type == AgentEvent.ERROR:
            st.session_state[f"{agent_name}_status"] = "Error"
            st.session_state.error_message = data['error']

class AgentManager:
    def __init__(self, event_handler: StreamlitEventHandler):
        self.event_handler = event_handler
        
    def run_agent(self, agent_name: str, *args, **kwargs):
        """Run an agent with event emission"""
        try:
            self.event_handler.handle_event(agent_name, AgentEvent.STARTED)
            result = self._execute_agent(agent_name, *args, **kwargs)
            self.event_handler.handle_event(agent_name, AgentEvent.COMPLETED, {'state': result})
        except Exception as e:
            self.event_handler.handle_event(agent_name, AgentEvent.ERROR, {'error': str(e)})
```

### 4. Wiki Generation Pipeline Integration

The wiki generation pipeline will be integrated through a wrapper class that manages the async operation and state updates:

```python
class StreamlitWikiGenerator:
    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        
    async def generate_wiki(self, topic: str):
        """Main wiki generation pipeline for Streamlit"""
        # 1. Initialize session state
        st.session_state.current_topic = topic
        st.session_state.wiki_state = {}
        
        # 2. Sequential agent execution with state updates
        try:
            # Research Phase
            research_result = await self.agent_manager.run_agent(
                "research_agent",
                topic=topic
            )
            
            # Planning Phase
            plan_result = await self.agent_manager.run_agent(
                "planner_agent",
                research=research_result
            )
            
            # Parallel Content & Design Phase
            content_task = self.agent_manager.run_agent(
                "content_writer_agent",
                plan=plan_result
            )
            design_task = self.agent_manager.run_agent(
                "design_coder_agent",
                plan=plan_result
            )
            
            content_result, design_result = await asyncio.gather(
                content_task,
                design_task
            )
            
            # Final merge and update
            final_wiki = self._merge_results(content_result, design_result)
            st.session_state.wiki_state = final_wiki
            
        except Exception as e:
            st.session_state.error_message = str(e)
            raise
```

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
│   └── State File Upload
├── Main Content
│   ├── Topic Input
│   │   ├── Text Input
│   │   └── Generate Button
│   └── Wiki Display
│       ├── Table of Contents
│       └── Content Sections
└── Error Display (if any)
```

## Notes

- Using default Streamlit theming for modern look
- Single page design for simplicity
- Read-only content during generation
- No authentication required
- Native Streamlit features preferred over custom implementations
- Error handling limited to clear message display
