"""State management utilities for the Streamlit UI."""

import json
from pathlib import Path
import streamlit as st


def initialize_session_state():
    """Initialize the session state with default values if not already set."""
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = ""

    if "wiki_path" not in st.session_state:
        st.session_state.wiki_path = None

    if "error_message" not in st.session_state:
        st.session_state.error_message = ""


def clear_error():
    """Clear any error message in the session state."""
    st.session_state.error_message = ""
