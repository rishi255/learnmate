import json
from pathlib import Path
from typing import Any, Dict

import yaml

from backend.paths import CONFIG_FILE_PATH, OUTPUTS_DIR


def load_config(config_path: str = CONFIG_FILE_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cleaned_topic_name(topic: str) -> str:
    """Cleans the topic name by removing unwanted characters.

    Args:
        topic: The raw topic name

    Returns:
        str: Cleaned topic name suitable for file paths
    """
    return "".join(
        char for char in topic if char.isalnum() or char in ("_", " ") or char.isspace()
    ).replace(" ", "_")


def get_topic_directory_name(topic: str, create: bool = True) -> str:
    """Cleans topic name and returns (and optionally creates) the directory path for storing topic outputs.

    Args:
        topic: The topic name. This can be raw, it will get cleaned
        create: Whether to create the directory if it doesn't exist

    Returns:
        str: Absolute path to the topic directory
    """
    topic_name = get_cleaned_topic_name(topic)
    topic_dir = Path(OUTPUTS_DIR) / topic_name

    if create:
        topic_dir.mkdir(parents=True, exist_ok=True)

    return str(topic_dir)


def save_state_checkpoint(state: Dict[str, Any], topic_dir: str) -> None:
    """Save the current state as a JSON checkpoint.

    Args:
        state: Current state dictionary
        topic_dir: Directory for the current topic

    The state is saved to saved_wiki_state.json in the topic directory.
    If the file exists, it is overwritten with the new state.
    """
    checkpoint_path = Path(topic_dir) / "saved_wiki_state.json"

    # Convert state to JSON-serializable format
    state_json = json.dumps(state, indent=4, default=str)

    # Save to file
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        f.write(state_json)
