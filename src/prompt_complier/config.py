"""
Configuration management for the Prompt Compiler.

This module initializes the `dynaconf` settings object, which loads configuration
from `settings.toml`, `.secrets.toml`, and environment variables. It serves as the
central point for accessing application settings.
"""

from pathlib import Path

from dynaconf import Dynaconf

# Resolve the project root directory
# src/prompt_complier/config.py -> src/prompt_complier -> src -> root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SETTINGS_FILE = BASE_DIR / "settings.toml"
SECRETS_FILE = BASE_DIR / ".secrets.toml"

# Global settings object initialized with Dynaconf
settings = Dynaconf(
    envvar_prefix="PRCOMP",
    settings_files=[str(SETTINGS_FILE), str(SECRETS_FILE)],
    environments=True,
    env_switcher="PRCOMP_MODE",
    load_dotenv=True,
    merge_enabled=True,
    validators=[
        # Add validators here to ensure critical configuration is present
        # Validator("API_KEY", must_exist=True),
    ],
)

# Trigger validation
settings.validators.validate()
