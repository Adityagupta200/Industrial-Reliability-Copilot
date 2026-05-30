import pytest

from llm_orchestrator.guardrails.output_filters import OutputGuardrails


def test_deterministic_groundedness_passes_cited_supported_answer() -> None:
    context = (
        "[DOC_1]\n"
        "High vibration with stable pressure and flow is a common indicator of "
        "bearing wear, insufficient lubrication, contamination, or misalignment."
    )
    initial_input = (
        "Sensor Data: vibration RMS 8.4, pressure 5.2 bar, flow 176.0 lpm, "
        "no corresponding pressure drop."
    )
    answer = """
    {
      "hypotheses": [
        {
          "cause": "Bearing wear or insufficient lubrication",
          "confidence": 0.9,
          "evidence": "DOC_1 states high vibration with stable pressure and flow supports bearing wear or insufficient lubrication. Sensor readings show vibration RMS 8.4 with stable pressure and flow.",
          "source": "DOC_1"
        }
      ]
    }
    """

    assert OutputGuardrails._deterministic_groundedness(context, answer, initial_input) == 1.0


def test_deterministic_groundedness_blocks_hallucinated_doc_tag() -> None:
    context = "[DOC_1]\nInspect bearing housing for contamination."
    answer = '{"hypotheses": [{"evidence": "DOC_9 supports this.", "source": "DOC_9"}]}'

    assert OutputGuardrails._deterministic_groundedness(context, answer) == 0.0


@pytest.mark.asyncio
async def test_groundedness_fast_path_does_not_call_llm_judge() -> None:
    class FailingLLM:
        async def invoke(self, *args, **kwargs):  # pragma: no cover - should never be reached
            raise AssertionError("LLM judge should not be called for deterministic pass")

    context = (
        "[DOC_1]\n"
        "High vibration with stable pressure and flow is a common indicator of "
        "bearing wear and insufficient lubrication."
    )
    answer = (
        '{"hypotheses":[{"cause":"Bearing wear","confidence":0.9,'
        '"evidence":"DOC_1 links high vibration with stable pressure and flow '
        'to bearing wear and insufficient lubrication.","source":"DOC_1"}]}'
    )

    score = await OutputGuardrails.check_groundedness(FailingLLM(), context, answer)

    assert score == 1.0


@pytest.mark.asyncio
async def test_groundedness_audit_mode_records_real_llm_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTPUT_GUARDRAILS_LLM_JUDGE_MODE", "audit")

    class PassingJudge:
        calls = 0

        async def invoke(self, prompt: str, **kwargs):
            self.calls += 1
            assert "Request Evidence" in prompt
            assert kwargs["is_judge"] is True
            assert kwargs["force_provider"] == "openai"

            class Result:
                content = "PASS"

            return Result()

    llm = PassingJudge()
    context = (
        "[DOC_1]\n"
        "High vibration with stable pressure and flow is a common indicator of "
        "bearing wear and insufficient lubrication."
    )
    initial_input = "Sensor Data: vibration RMS 8.4, pressure 5.2 bar, flow 176.0 lpm."
    answer = (
        '{"hypotheses":[{"cause":"Bearing wear","confidence":0.9,'
        '"evidence":"DOC_1 links high vibration with stable pressure and flow '
        'to bearing wear and insufficient lubrication while sensor data shows vibration RMS 8.4.",'
        '"source":"DOC_1"}]}'
    )

    score = await OutputGuardrails.check_groundedness(llm, context, answer, initial_input)

    assert score == 1.0
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_groundedness_strict_mode_enforces_llm_judge_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTPUT_GUARDRAILS_LLM_JUDGE_MODE", "strict")

    class FailingJudge:
        async def invoke(self, *args, **kwargs):
            class Result:
                content = "FAIL"

            return Result()

    context = (
        "[DOC_1]\n"
        "High vibration with stable pressure and flow is a common indicator of "
        "bearing wear and insufficient lubrication."
    )
    answer = (
        '{"hypotheses":[{"cause":"Bearing wear","confidence":0.9,'
        '"evidence":"DOC_1 links high vibration with stable pressure and flow '
        'to bearing wear and insufficient lubrication.","source":"DOC_1"}]}'
    )

    score = await OutputGuardrails.check_groundedness(FailingJudge(), context, answer)

    assert score == 0.0
