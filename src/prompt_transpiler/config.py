"""
Configuration management for the Prompt Transpiler.

This module initializes the `dynaconf` settings object, which loads configuration
from `settings.toml`, `.secrets.toml`, and environment variables. It serves as the
central point for accessing application settings.
"""

import os
from pathlib import Path

from dynaconf import Dynaconf

# Resolve the project root directory
# src/prompt_transpiler/config.py -> src/prompt_transpiler -> src -> root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SETTINGS_FILE = BASE_DIR / "settings.toml"
SECRETS_FILE = BASE_DIR / ".secrets.toml"

# Mirror legacy environment variable names for compatibility during the rename.
for legacy_key, value in list(os.environ.items()):
    if legacy_key.startswith("PRCOMP_"):
        new_key = legacy_key.replace("PRCOMP_", "PRTRANS_", 1)
        os.environ.setdefault(new_key, value)
if "PRCOMP_MODE" in os.environ:
    os.environ.setdefault("PRTRANS_MODE", os.environ["PRCOMP_MODE"])

# Global settings object initialized with Dynaconf
settings = Dynaconf(
    envvar_prefix="PRTRANS",
    settings_files=[str(SETTINGS_FILE), str(SECRETS_FILE)],
    environments=True,
    env_switcher="PRTRANS_MODE",
    load_dotenv=True,
    merge_enabled=True,
    validators=[
        # Add validators here to ensure critical configuration is present
        # Validator("API_KEY", must_exist=True),
    ],
)

# Trigger validation
settings.validators.validate()

if not settings.get("TRANSPILER"):
    compiler_settings = settings.get("COMPILER")
    if compiler_settings is not None:
        settings.set("TRANSPILER", compiler_settings)
