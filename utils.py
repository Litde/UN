import json
import os

EXPECTED_KEYS = {"COMET_API_KEY", "COMET_PROJECT_NAME", "COMET_WORKSPACE_NAME"}

def load_comet_credentials(path: str) -> dict:
    """
    Load and validate a credentials JSON file with fields:
    {
        "COMET_API_KEY": "",
        "COMET_PROJECT_NAME": "",
        "COMET_WORKSPACE_NAME": ""
    }

    Returns:
        dict: The parsed credentials.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is invalid or keys are missing/extra.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Credentials file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"File is not valid JSON: {e}")

    data_keys = set(data.keys())

    missing = EXPECTED_KEYS - data_keys
    extra = data_keys - EXPECTED_KEYS

    if missing:
        raise ValueError(f"Credentials file is missing keys: {missing}")
    if extra:
        raise ValueError(f"Credentials file contains unexpected keys: {extra}")

    for key in EXPECTED_KEYS:
        if not isinstance(data[key], str):
            raise ValueError(f"Key '{key}' must be a string, got {type(data[key]).__name__}")

    return data
