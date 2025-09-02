"""Wiki viewer component for displaying the generated wiki content."""

import streamlit as st
from pathlib import Path


def display_wiki_content(wiki_path: str):
    """Display the wiki content from the markdown file."""
    if not wiki_path or not Path(wiki_path).exists():
        return

    # Read and display the markdown file
    with open(wiki_path, 'r') as f:
        content = f.read()
        st.markdown(content)
