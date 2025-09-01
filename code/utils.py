import json
import yaml
from pathlib import Path
from paths import CONFIG_FILE_PATH
from typing import Dict, Any


def load_config(config_path: str = CONFIG_FILE_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
