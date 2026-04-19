from prompt_transpiler.config import settings


def test_dynaconf_env_override(monkeypatch):
    """Test that environment variables override settings."""
    # Dynaconf is configured with envvar_prefix="PRTRANS"

    # 1. Test a new key
    monkeypatch.setenv("PRTRANS_TEST_KEY", "test_value")

    # Force reload to pick up env vars
    settings.reload()

    assert settings.TEST_KEY == "test_value"


def test_dynaconf_nested_override(monkeypatch):
    """Test overriding nested settings."""
    # Assuming we want to override OPENAI.API_KEY -> PRTRANS_OPENAI__API_KEY
    monkeypatch.setenv("PRTRANS_OPENAI__API_KEY", "sk-test-key")

    settings.reload()

    # Check if it's accessible via attribute access
    assert settings.OPENAI.API_KEY == "sk-test-key"
