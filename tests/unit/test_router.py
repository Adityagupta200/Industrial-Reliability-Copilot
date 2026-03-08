from llm_orchestrator.router import heuristic_route


def test_router_historical():
    assert heuristic_route("show similar incidents in last 6 months") == "historical"


def test_router_remediation():
    assert heuristic_route("how to fix bearing wear procedure") == "remediation"


def test_router_root_cause_default():
    assert heuristic_route("why did pump P-23 trigger anomaly") == "root_cause"
