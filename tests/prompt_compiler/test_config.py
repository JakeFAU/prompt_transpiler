from prompt_complier.config import settings


def test_dynaconf_env_override(monkeypatch):
    """Test that environment variables override settings."""
    # Dynaconf is configured with envvar_prefix="PRCOMP"

    # 1. Test a new key
    monkeypatch.setenv("PRCOMP_TEST_KEY", "test_value")

    # Force reload to pick up env vars
    settings.reload()

    assert settings.TEST_KEY == "test_value"


def test_dynaconf_nested_override(monkeypatch):
    """Test overriding nested settings."""
    # Assuming we want to override OPENAI.API_KEY -> PRCOMP_OPENAI__API_KEY
    monkeypatch.setenv("PRCOMP_OPENAI__API_KEY", "sk-test-key")

    settings.reload()

    # Check if it's accessible via attribute access
    assert settings.OPENAI.API_KEY == "sk-test-key"
