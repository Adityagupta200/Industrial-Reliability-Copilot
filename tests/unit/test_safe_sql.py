import pytest
from llm_orchestrator.db.safe_sql import validate_readonly_sql, SQLPolicy, UnsafeSQLError


def test_reject_non_select():
    with pytest.raises(UnsafeSQLError):
        validate_readonly_sql("DELETE FROM incidents", SQLPolicy(allowed_tables={"incidents"}))


def test_enforces_limit_and_table():
    sql = "SELECT id, timestamp FROM incidents"
    out = validate_readonly_sql(sql, SQLPolicy(allowed_tables={"incidents"}, max_limit=100))
    assert "LIMIT" in out.upper()


def test_rejects_other_tables():
    with pytest.raises(UnsafeSQLError):
        validate_readonly_sql("SELECT * FROM users", SQLPolicy(allowed_tables={"incidents"}))
