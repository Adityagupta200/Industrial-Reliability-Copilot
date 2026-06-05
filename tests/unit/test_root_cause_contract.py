import pytest

from llm_orchestrator.chains.root_cause_chain import (
    _build_general_fast_path,
    _build_supported_bearing_fast_path,
    _dedupe_hypotheses,
    _fast_path_guardrail_answer,
    _format_docs,
    _is_general_fast_path_candidate,
    _is_high_vibration_bearing_case,
    _prefer_direct_procedure_fast_path,
    _stabilize_supported_bearing_hypothesis,
    _validate_fast_path_response,
    RootCauseChain,
)
from llm_orchestrator.guardrails.output_filters import OutputGuardrails
from llm_orchestrator.schemas import Hypothesis, RetrievedDoc, RootCauseRequest, RootCauseResponse


def test_stabilize_supported_bearing_hypothesis_preserves_grounded_contract() -> None:
    response = RootCauseResponse(
        hypotheses=[
            Hypothesis(
                cause="Mechanical imbalance",
                confidence=0.55,
                evidence="High vibration is discussed in the cited procedure.",
                source="bearing_replacement_pump_P-23.md",
            )
        ]
    )
    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger anomaly at 03:41?",
        anomaly_description=(
            "Pump P-23 triggered a high-vibration anomaly with no corresponding pressure drop."
        ),
        equipment_id="pump_P-23",
        sensor_data={"vibration_rms": 8.4, "pressure_bar": 5.2, "flow_rate_lpm": 176.0},
    )
    docs_text = (
        "[DOC_1]\nHigh vibration with stable pressure and flow is a common indicator of "
        "bearing wear, insufficient lubrication, contamination, or misalignment."
    )

    stabilized = _stabilize_supported_bearing_hypothesis(
        response,
        request,
        docs_text,
        {"DOC_1": "bearing_replacement_pump_P-23.md"},
    )

    primary = stabilized.hypotheses[0]
    assert primary.cause == "Bearing wear or insufficient lubrication"
    assert "bearing wear" in primary.evidence.lower()
    assert "insufficient lubrication" in primary.evidence.lower()
    assert primary.source == "bearing_replacement_pump_P-23.md"
    assert primary.confidence >= 0.72


def test_stabilize_supported_bearing_hypothesis_does_not_invent_without_context() -> None:
    response = RootCauseResponse(
        hypotheses=[
            Hypothesis(
                cause="Cavitation",
                confidence=0.6,
                evidence="Pressure instability is discussed in the cited procedure.",
                source="cavitation_triage_pump.md",
            )
        ]
    )
    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger anomaly at 03:41?",
        anomaly_description="Pump P-23 triggered a pressure anomaly.",
        equipment_id="pump_P-23",
        sensor_data={"vibration_rms": 1.2, "pressure_bar": 0.5},
    )

    stabilized = _stabilize_supported_bearing_hypothesis(
        response,
        request,
        "[DOC_1]\nInspect suction blockage and cavitation.",
    )

    assert stabilized.hypotheses[0].cause == "Cavitation"


def test_dedupe_hypotheses_removes_repeated_bearing_lubrication_topic() -> None:
    response = RootCauseResponse(
        hypotheses=[
            Hypothesis(
                cause="Bearing wear or insufficient lubrication",
                confidence=0.86,
                evidence="High vibration and stable pressure indicate bearing wear.",
                source="bearing_replacement_pump_P-23.md",
            ),
            Hypothesis(
                cause="Insufficient lubrication or contaminated grease",
                confidence=0.74,
                evidence="Grease starvation can cause bearing wear.",
                source="bearing_replacement_pump_P-23.md",
            ),
            Hypothesis(
                cause="Coupling or shaft misalignment",
                confidence=0.58,
                evidence="Alignment checks are required when vibration rises.",
                source="pump_manual2.pdf",
            ),
        ]
    )

    deduped = _dedupe_hypotheses(response)

    assert [h.cause for h in deduped.hypotheses] == [
        "Bearing wear or insufficient lubrication",
        "Coupling or shaft misalignment",
    ]


def test_format_docs_applies_context_budget_without_losing_source_mapping() -> None:
    docs_text = "High vibration with stable pressure indicates bearing wear. " * 20
    formatted, mapping = _format_docs(
        [
            RetrievedDoc(
                id="doc-1",
                text=docs_text,
                metadata={"source_file": "bearing_replacement_pump_P-23.md"},
                score=0.9,
                source="semantic",
            )
        ],
        max_chars_per_doc=160,
    )

    assert mapping == {"DOC_1": "bearing_replacement_pump_P-23.md"}
    assert "[DOC_1]" in formatted
    assert "[TRUNCATED:" in formatted
    assert len(formatted) < len(docs_text)


def test_supported_bearing_fast_path_returns_grounded_response(monkeypatch) -> None:
    monkeypatch.setenv("ROOT_CAUSE_FAST_PATH_ENABLED", "true")
    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger anomaly at 03:41?",
        anomaly_description=(
            "Pump P-23 triggered a high-vibration anomaly with no corresponding pressure drop."
        ),
        equipment_id="pump_P-23",
        sensor_data={"vibration_rms": 8.4, "pressure_bar": 5.2, "flow_rate_lpm": 176.0},
    )
    docs_text = (
        "[DOC_1]\n"
        "High vibration with stable pressure and flow is a common indicator of "
        "bearing wear, insufficient lubrication, contamination, or misalignment. "
        "Perform an alignment check after bearing replacement."
    )

    response = _build_supported_bearing_fast_path(
        request,
        docs_text,
        {"DOC_1": "bearing_replacement_pump_P-23.md"},
    )

    assert response is not None
    assert response.hypotheses[0].cause == "Bearing wear or insufficient lubrication"
    assert response.hypotheses[0].source == "bearing_replacement_pump_P-23.md"
    assert response.hypotheses[0].confidence == 0.9
    assert "vibration RMS 8.4" in response.hypotheses[0].evidence
    assert [hypothesis.cause for hypothesis in response.hypotheses] == [
        "Bearing wear or insufficient lubrication",
        "Pump or coupling misalignment",
    ]


def test_supported_bearing_fast_path_accepts_manual_bearing_vibration_evidence(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_CAUSE_FAST_PATH_ENABLED", "true")
    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger a high-vibration anomaly at 03:41?",
        anomaly_description="Pump P-23 has vibration RMS 8.4 with stable pressure and flow.",
        equipment_id="pump_P-23",
        sensor_data={
            "vibration_rms": 8.4,
            "temp_c": 74.2,
            "pressure_bar": 5.2,
            "flow_rate_lpm": 176.0,
        },
    )
    docs_text = (
        "[DOC_1]\n"
        "Pump operating guidance: bearing temperature and vibrations must be monitored "
        "to mitigate malfunction risk. Excessive vibrations indicate uneven pump "
        "operation and the bearing has a minimum design life before replacement."
    )

    response = _build_supported_bearing_fast_path(
        request,
        docs_text,
        {"DOC_1": "pump_manual2.pdf"},
    )

    assert response is not None
    assert response.hypotheses[0].cause == "Bearing wear or damage causing excessive vibration"
    assert response.hypotheses[0].source == "pump_manual2.pdf"
    assert "vibration RMS 8.4" in response.hypotheses[0].evidence


def test_general_fast_path_routes_turbofan_vibration_before_generic_overheating(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_CAUSE_FAST_PATH_ENABLED", "true")
    request = RootCauseRequest(
        user_query="Why did turbofan TF-01 show high vibration during cruise-speed simulation?",
        anomaly_description=(
            "Turbofan TF-01 has elevated vibration with normal flow and no pressure collapse."
        ),
        equipment_id="turbofan_TF-01",
        sensor_data={
            "vibration_rms": 8.0,
            "temp_c": 87.0,
            "pressure_bar": 8.1,
            "flow_rate_lpm": 181.0,
        },
    )
    docs_text = (
        "[DOC_1]\n"
        "Engine maintenance guidance: high turbofan vibration requires bearing "
        "condition checks, rotating-assembly inspection, and service verification."
    )

    response = _build_general_fast_path(
        request,
        docs_text,
        {"DOC_1": "bearing_installation_manuals.pdf"},
    )

    assert _is_high_vibration_bearing_case(request) is False
    assert _is_general_fast_path_candidate(request) is True
    assert response is not None
    assert response.hypotheses[0].cause == (
        "Engine rotating-assembly vibration requiring bearing inspection"
    )
    assert "bearing" in response.hypotheses[0].evidence.lower()
    assert "overheating from load" not in response.hypotheses[0].cause.lower()


def test_general_fast_path_keeps_oil_quality_ahead_of_turbofan_branch(monkeypatch) -> None:
    monkeypatch.setenv("ROOT_CAUSE_FAST_PATH_ENABLED", "true")
    request = RootCauseRequest(
        user_query="Could low oil quality explain TF-01 vibration and temperature drift?",
        anomaly_description=(
            "Vibration and temperature rose gradually after oil filter differential "
            "pressure increased."
        ),
        equipment_id="turbofan_TF-01",
        sensor_data={
            "vibration_rms": 3.4,
            "temp_c": 111.0,
            "pressure_bar": 3.7,
            "flow_rate_lpm": 72.0,
        },
    )
    docs_text = (
        "[DOC_1]\n"
        "Lubrication inspection guidance: check oil quality, oil filter restriction, "
        "lubrication intervals, and bearing condition when vibration and temperature "
        "drift together."
    )

    response = _build_general_fast_path(
        request,
        docs_text,
        {"DOC_1": "bearing_installation_manuals.pdf"},
    )

    assert response is not None
    assert response.hypotheses[0].cause == "Lubrication or oil-filter restriction"
    assert "oil filter" in response.hypotheses[0].evidence.lower()


def test_general_fast_path_prefers_direct_sops_only_when_a_matching_sop_exists() -> None:
    cavitation_request = RootCauseRequest(
        user_query="Why is pump P-23 making gravel-like noise with low flow?",
        anomaly_description="Reduced flow and fluctuating discharge pressure indicate cavitation.",
        equipment_id="pump_P-23",
        sensor_data={"pressure_bar": 0.8, "flow_rate_lpm": 93.0},
    )
    turbofan_request = RootCauseRequest(
        user_query="Why did turbofan TF-01 show high vibration?",
        anomaly_description="Turbofan TF-01 has elevated vibration.",
        equipment_id="turbofan_TF-01",
        sensor_data={"vibration_rms": 8.0, "temp_c": 87.0},
    )
    oil_request = RootCauseRequest(
        user_query="Could low oil quality explain TF-01 vibration and temperature drift?",
        anomaly_description="Oil filter differential pressure increased.",
        equipment_id="turbofan_TF-01",
        sensor_data={"temp_c": 111.0},
    )

    assert _prefer_direct_procedure_fast_path(cavitation_request) is True
    assert _prefer_direct_procedure_fast_path(turbofan_request) is False
    assert _prefer_direct_procedure_fast_path(oil_request) is False


def test_fast_path_guardrail_payload_preserves_doc_tag_grounding() -> None:
    response = RootCauseResponse(
        hypotheses=[
            Hypothesis(
                cause="Bearing wear or insufficient lubrication",
                confidence=0.9,
                evidence=(
                    "The bearing_replacement_pump_P-23.md states that high vibration "
                    "with stable pressure and flow supports bearing wear."
                ),
                source="bearing_replacement_pump_P-23.md",
            )
        ]
    )
    context = (
        "[DOC_1]\n"
        "High vibration with stable pressure and flow is a common indicator of "
        "bearing wear and insufficient lubrication."
    )

    answer = _fast_path_guardrail_answer(
        response,
        {"DOC_1": "bearing_replacement_pump_P-23.md"},
    )

    assert '"source": "DOC_1"' in answer
    assert "DOC_1 supports" in answer
    assert OutputGuardrails._deterministic_groundedness(context, answer) == 1.0


@pytest.mark.asyncio
async def test_fast_path_validation_uses_deterministic_grounding_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTPUT_GUARDRAILS_LLM_JUDGE_MODE", "fallback")

    class FailingLLM:
        async def invoke(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("Fast-path fallback validation should not call the LLM judge")

    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger anomaly at 03:41?",
        anomaly_description=(
            "Pump P-23 triggered a high-vibration anomaly with no corresponding pressure drop."
        ),
        equipment_id="pump_P-23",
        sensor_data={"vibration_rms": 8.4, "pressure_bar": 5.2, "flow_rate_lpm": 176.0},
    )
    response = RootCauseResponse(
        hypotheses=[
            Hypothesis(
                cause="Bearing wear or insufficient lubrication",
                confidence=0.9,
                evidence=(
                    "The bearing_replacement_pump_P-23.md states that high vibration "
                    "with stable pressure and flow supports bearing wear or insufficient "
                    "lubrication. Sensor readings show vibration RMS 8.4 with stable "
                    "pressure and flow."
                ),
                source="bearing_replacement_pump_P-23.md",
            )
        ]
    )
    context = (
        "[DOC_1]\n"
        "High vibration with stable pressure and flow is a common indicator of "
        "bearing wear and insufficient lubrication."
    )

    await _validate_fast_path_response(
        llm_client=FailingLLM(),
        req=request,
        response=response,
        docs_text=context,
        doc_mapping={"DOC_1": "bearing_replacement_pump_P-23.md"},
        anomaly_model={"anomaly": {"description": "High vibration with stable pressure."}},
    )


@pytest.mark.asyncio
async def test_high_vibration_chain_uses_direct_sop_before_heavier_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOT_CAUSE_FAST_PATH_ENABLED", "true")
    monkeypatch.setenv("OUTPUT_GUARDRAILS_LLM_JUDGE_MODE", "fallback")

    class FailingLLM:
        async def invoke(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("Direct SOP fast path should not call the LLM")

    class FakeAnomalyClient:
        async def predict(self, sensor_data):
            return {"anomaly": {"description": "High vibration with stable pressure."}}

    class FakeRAGClient:
        async def retrieve_procedures_direct(self, query, *, equipment_id=None, k=5):
            return [
                RetrievedDoc(
                    id="procedure:bearing",
                    text=(
                        "High vibration with stable pressure and flow is a common "
                        "indicator of bearing wear, insufficient lubrication, "
                        "contamination, or misalignment."
                    ),
                    metadata={"source_file": "bearing_replacement_pump_P-23.md"},
                    source="keyword",
                    score=0.95,
                )
            ]

        async def retrieve_procedures(
            self,
            failure_mode,
            *,
            equipment_id=None,
            k=6,
            query=None,
        ):
            raise AssertionError("Semantic procedure retrieval should not run")

        async def retrieve_hybrid(self, query, *, equipment_id=None, k=8):
            raise AssertionError("Hybrid retrieval should not run")

    chain = RootCauseChain(
        llm=FailingLLM(),
        prompts=object(),
        anomaly_client=FakeAnomalyClient(),
        rag_client=FakeRAGClient(),
    )
    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger a high-vibration anomaly at 03:41?",
        anomaly_description="Pump P-23 has vibration RMS 8.4 with stable pressure and flow.",
        equipment_id="pump_P-23",
        sensor_data={
            "vibration_rms": 8.4,
            "temp_c": 74.2,
            "pressure_bar": 5.2,
            "flow_rate_lpm": 176.0,
        },
    )

    response, provider, model, raw_context = await chain.run(request)

    assert provider == "rules+retrieval"
    assert model == "root-cause-fast-path-v1"
    assert response.hypotheses[0].cause == "Bearing wear or insufficient lubrication"
    assert response.hypotheses[0].source == "bearing_replacement_pump_P-23.md"
    assert "[DOC_1]" in raw_context


@pytest.mark.asyncio
async def test_high_vibration_chain_uses_manual_backfill_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOT_CAUSE_FAST_PATH_ENABLED", "true")

    class FailingLLM:
        async def invoke(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("Manual-backed fast path should not call the LLM")

    class FakeAnomalyClient:
        async def predict(self, sensor_data):
            return {"anomaly": {"description": "High vibration with stable pressure."}}

    class FakeRAGClient:
        async def retrieve_procedures_direct(self, query, *, equipment_id=None, k=5):
            return [
                RetrievedDoc(
                    id="procedure:generic",
                    text="Inspect the asset and document observations.",
                    metadata={"source_file": "scheduled_maintenance_template.md"},
                    source="keyword",
                    score=0.1,
                )
            ]

        async def retrieve_procedures(
            self,
            failure_mode,
            *,
            equipment_id=None,
            k=6,
            query=None,
        ):
            return []

        async def retrieve_hybrid(self, query, *, equipment_id=None, k=8):
            return [
                RetrievedDoc(
                    id="manual:pump-bearing",
                    text=(
                        "High vibration with stable pressure and flow is a common "
                        "indicator of bearing wear, insufficient lubrication, "
                        "contamination, or misalignment."
                    ),
                    metadata={"source_file": "pump_manual2.pdf", "equipment_id": "pump_P-23"},
                    source="hybrid",
                    score=0.9,
                )
            ]

    chain = RootCauseChain(
        llm=FailingLLM(),
        prompts=object(),
        anomaly_client=FakeAnomalyClient(),
        rag_client=FakeRAGClient(),
    )
    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger a high-vibration anomaly at 03:41?",
        anomaly_description="Pump P-23 has vibration RMS 8.4 with stable pressure and flow.",
        equipment_id="pump_P-23",
        sensor_data={
            "vibration_rms": 8.4,
            "temp_c": 74.2,
            "pressure_bar": 5.2,
            "flow_rate_lpm": 176.0,
        },
    )

    response, provider, model, raw_context = await chain.run(request)

    assert provider == "rules+retrieval"
    assert model == "root-cause-fast-path-v1"
    assert response.hypotheses[0].cause == "Bearing wear or insufficient lubrication"
    assert response.hypotheses[0].source == "pump_manual2.pdf"
    assert "[DOC_1]" in raw_context


@pytest.mark.asyncio
async def test_general_fast_path_uses_procedure_vector_fallback_before_hybrid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOT_CAUSE_FAST_PATH_ENABLED", "true")

    class FailingLLM:
        async def invoke(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("Procedure-backed fast path should not call the LLM")

    class FakeAnomalyClient:
        async def predict(self, sensor_data):
            return {"anomaly": {"description": "Low pressure and reduced flow."}}

    class FakeRAGClient:
        async def retrieve_procedures_direct(self, query, *, equipment_id=None, k=5):
            return []

        async def retrieve_procedures(
            self,
            failure_mode,
            *,
            equipment_id=None,
            k=6,
            query=None,
        ):
            return [
                RetrievedDoc(
                    id="procedure:cavitation",
                    text=(
                        "Cavitation triage: check suction strainer for blockage, "
                        "verify NPSH conditions, inspect for air ingress, and monitor "
                        "fluctuating discharge pressure with reduced flow."
                    ),
                    metadata={"source_file": "cavitation_triage_pump.md"},
                    source="semantic",
                    score=0.9,
                )
            ]

        async def retrieve_hybrid(self, query, *, equipment_id=None, k=8):
            raise AssertionError("Hybrid retrieval should not run when procedure vectors match")

    chain = RootCauseChain(
        llm=FailingLLM(),
        prompts=object(),
        anomaly_client=FakeAnomalyClient(),
        rag_client=FakeRAGClient(),
    )
    request = RootCauseRequest(
        user_query=(
            "Why is pump P-23 making gravel-like noise with low flow and "
            "fluctuating discharge pressure?"
        ),
        anomaly_description=(
            "Gravel-like noise, reduced flow, and discharge pressure oscillation "
            "indicate possible cavitation."
        ),
        equipment_id="pump_P-23",
        sensor_data={
            "vibration_rms": 5.9,
            "temp_c": 65.0,
            "pressure_bar": 0.8,
            "flow_rate_lpm": 93.0,
        },
    )

    response, provider, model, raw_context = await chain.run(request)

    assert provider == "rules+retrieval"
    assert model == "root-cause-general-fast-path-v1"
    assert response.hypotheses[0].cause == "Cavitation or suction-side restriction"
    assert response.hypotheses[0].source == "cavitation_triage_pump.md"
    assert "[DOC_1]" in raw_context
