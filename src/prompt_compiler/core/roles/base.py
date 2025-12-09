from typing import Any


class BaseRole:
    """
    Base class for all agentic roles (Architect, Pilot, Judge).
    Provides standardized OTel attributes.
    """

    @property
    def role_name(self) -> str:
        """
        Returns the role name.
        Subclasses should override this if the class name is not appropriate.
        Defaults to the class name in lowercase.
        """
        return self.__class__.__name__.lower()

    def _get_base_attributes(self) -> dict[str, Any]:
        """Standard attributes for all roles."""
        # These attributes are expected to be present on the concrete classes
        # via attrs or manual definition.
        model_name = getattr(self, "model_name", "unknown")
        provider_name = getattr(self, "provider_name", "unknown")

        return {
            "prompt_compiler.role": self.role_name,
            "gen_ai.system": provider_name,
            "gen_ai.request.model": model_name,
        }
