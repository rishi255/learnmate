"""Sidebar components for the Streamlit UI."""

import streamlit as st
import json
from pathlib import Path


def setup_sidebar():
    """Setup the sidebar with title and state file upload.
    Returns the state file path if a valid state file was uploaded, None otherwise."""
    st.sidebar.title("LearnMate")
    st.sidebar.markdown("Visual Wiki Generator")

    # State file upload
    uploaded_file = st.sidebar.file_uploader(
        "Resume from state file",
        type=["json"],
        help="Upload a previously saved state file to resume generation",
    )

    if uploaded_file is not None:
        try:
            # Create temp directory for uploaded files if it doesn't exist
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)

            # Save uploaded file to temp directory
            temp_file_path = temp_dir / uploaded_file.name
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Verify it's valid JSON
            with open(temp_file_path, "r") as f:
                json.load(f)

            st.sidebar.success("State file loaded successfully!")
            return str(temp_file_path.absolute())

        except json.JSONDecodeError:
            st.sidebar.error("Invalid state file format")
        except Exception as e:
            st.sidebar.error(f"Error loading state file: {str(e)}")

    return None
