"""Main Streamlit application for LearnMate."""

import streamlit as st
from pathlib import Path
import sys
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.main import main as generate_wiki
from frontend.components import wiki_viewer, sidebar
from frontend.utils import state


def main():
    """Main Streamlit application."""
    # Initialize session state
    state.initialize_session_state()

    # Setup sidebar and get starting state file path if uploaded
    starting_state_file_path = sidebar.setup_sidebar()

    st.title("Generate a Visual Wiki")

    # Continue from existing state if state file was uploaded
    if starting_state_file_path:
        with st.spinner("Generating wiki from saved state..."):
            try:
                st.info(
                    f"Loading state from uploaded file: '{Path(starting_state_file_path).name}'."
                )

                # Run wiki generation with the starting state file
                wiki_path = generate_wiki(
                    user_input=None, starting_state_file_path=starting_state_file_path
                )

                if wiki_path:
                    st.success("Wiki generated successfully from saved state!")
                    wiki_viewer.display_wiki_content(wiki_path)
                    print("\n[streamlit]:: ✅ Wiki displayed successfully!")
                else:
                    st.warning(
                        "No wiki path returned from generation. The state file might be invalid or incomplete."
                    )
            except Exception as e:
                st.session_state.error_message = str(e)
                st.error(f"Error generating wiki: {traceback.format_exc()}")

        if st.button("Generate New Wiki"):
            state.clear_session_state()
            st.rerun()
        return

    # Topic input for new wiki
    user_input = st.text_input(
        "Enter a topic",
        value=st.session_state.current_topic,
        placeholder="e.g., 'How DNS Works' or 'Basics of Machine Learning'",
        help="Enter any topic you want to learn about",
    )

    # Generate button
    if st.button("Generate Wiki", disabled=not user_input):
        with st.spinner("Generating your wiki... This might take a few minutes."):
            try:
                # Clear any previous error
                state.clear_error()

                # Run wiki generation and get the path to generated wiki
                wiki_path = generate_wiki(user_input=user_input)
                st.session_state.current_topic = user_input

                # Display the generated wiki
                if wiki_path:
                    wiki_viewer.display_wiki_content(wiki_path)
                    st.success("Wiki generated successfully!")

            except Exception as e:
                st.session_state.error_message = str(e)
                st.error(f"Error generating wiki: {traceback.format_exc()}")

    # Display error message if any
    if st.session_state.error_message:
        st.error(st.session_state.error_message)


if __name__ == "__main__":
    main()
