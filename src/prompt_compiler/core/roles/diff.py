"""Diff agent role for summarizing prompt changes."""

import json

from attrs import define

from prompt_compiler.core.interfaces import IDiffAgent
from prompt_compiler.core.roles.base import BaseRole
from prompt_compiler.llm.factory import get_llm_provider
from prompt_compiler.llm.prompts.prompt_objects import CandidatePrompt, OriginalPrompt
from prompt_compiler.utils.logging import get_logger
from prompt_compiler.utils.telemetry import telemetry
from prompt_compiler.utils.token_collector import token_collector

logger = get_logger(__name__)


@define
class SemanticDiffAgent(IDiffAgent, BaseRole):
    """
    Generates a semantic diff between the original prompt and the optimized candidate.

    Adds an optional rationale explaining why major changes help the target model.
    """

    provider_name: str = "openai"
    model_name: str = "gpt-4o-mini"

    @property
    def role_name(self) -> str:
        return "diff"

    def _build_prompts(
        self, original: OriginalPrompt, candidate: CandidatePrompt
    ) -> tuple[str, str]:
        system_prompt = (
            "You are a prompt migration explainer. "
            "Compare the starting prompt to the optimized prompt and describe the "
            "semantic differences. Highlight intent, tone, constraints, input/output "
            "handling, and structure. Prefer concise bullet points."
        )

        user_prompt = (
            "START PROMPT:\n"
            f"{original.prompt}\n\n"
            "OPTIMIZED PROMPT:\n"
            f"{candidate.prompt}\n\n"
            "Return a compact explanation of the most important differences and, when possible,"
            " why they improve compatibility with the target model."
        )
        return system_prompt, user_prompt

    async def summarize_diff(self, original: OriginalPrompt, candidate: CandidatePrompt) -> str:
        """Generate a human-readable summary of prompt differences."""
        attributes = self._get_base_attributes()
        attributes["prompt_compiler.target_model"] = candidate.model.model_name

        with telemetry.span(f"{self.role_name}.summarize_diff", attributes=attributes):
            system_prompt, user_prompt = self._build_prompts(original, candidate)
            provider = get_llm_provider(self.provider_name)

            schema = {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_differences": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["summary", "key_differences"],
            }

            response = None
            try:
                response = await provider.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_name=self.model_name,
                    config={"temperature": 0.2},
                    response_schema=schema,
                )
                token_collector.add(self.model_name, response.usage)

                data = json.loads(response.content)
                summary = (data.get("summary") or "").strip()
                differences = [str(item).strip() for item in data.get("key_differences", [])]
                rationale = (data.get("rationale") or "").strip()

                lines: list[str] = []
                if summary:
                    lines.append(summary)
                clean_differences = [d for d in differences if d]
                if clean_differences:
                    lines.append(
                        "Key differences:\n" + "\n".join(f"- {d}" for d in clean_differences)
                    )
                if rationale:
                    lines.append(f"Why it helps: {rationale}")

                candidate.diff_summary = "\n\n".join(lines).strip()
                return candidate.diff_summary

            except json.JSONDecodeError as exc:
                logger.error("Diff agent returned non-JSON response", error=str(exc))
                candidate.diff_summary = (response.content if response else "").strip()
                return candidate.diff_summary
            except Exception as exc:
                logger.error("Diff agent failed", error=str(exc))
                candidate.diff_summary = None
                return ""
