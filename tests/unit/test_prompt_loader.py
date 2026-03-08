import pytest
from llm_orchestrator.prompts.loader import PromptLoader, PromptNotFoundError


def test_load_all_prompts():
    loader = PromptLoader()
    for name in ["root_cause_analysis", "remediation_guidance", "historical_search"]:
        b = loader.load(name, "1.0")
        assert b.metadata["version"] == "1.0"
        assert len(b.template) > 20


def test_missing_version_raises():
    loader = PromptLoader()
    with pytest.raises(PromptNotFoundError):
        loader.load("root_cause_analysis", "9.9")
