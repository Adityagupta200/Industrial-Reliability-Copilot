from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

class UnsafeSQLError(ValueError):
    pass

@dataclass(frozen=True)
class SQLPolicy:
    allowed_tables: set[str]
    max_limit: int = 200

def validate_readonly_sql(sql: str, policy: SQLPolicy) -> str:
    try:
        parsed = sqlglot.parse_one(sql, read="postgres")
    except Exception as e:
        raise UnsafeSQLError(f"SQL parse failed: {e}") from e

    if not isinstance(parsed, exp.Select):
        raise UnsafeSQLError("Only SELECT queries are allowed.")

    # PRODUCTION FIX: Dynamically build the tuple of blocked types based on what 
    # exists in this specific version of sqlglot. This avoids both the 
    # AttributeError (if Truncate is missing) AND the TypeError (by passing valid types to find).
    blocked_names = ["Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Truncate"]
    blocked_types = tuple(getattr(exp, name) for name in blocked_names if hasattr(exp, name))
    
    # We unpack the tuple using *blocked_types so sqlglot receives them correctly
    if parsed.find(*blocked_types) is not None:
        raise UnsafeSQLError("Mutating/DDL SQL is not allowed.")

    # Ensure FROM tables are allowed
    tables = {t.name for t in parsed.find_all(exp.Table)}
    if not tables.issubset(policy.allowed_tables):
        raise UnsafeSQLError(f"SQL references disallowed tables: {tables - policy.allowed_tables}")

    # Enforce LIMIT
    limit = parsed.args.get("limit")
    if limit is None:
        parsed.set("limit", exp.Limit(this=exp.Literal.number(min(50, policy.max_limit))))
    else:
        try:
            lim_n = int(limit.expression.name)  # type: ignore[union-attr]
            if lim_n > policy.max_limit:
                parsed.set("limit", exp.Limit(this=exp.Literal.number(policy.max_limit)))
        except Exception:
            parsed.set("limit", exp.Limit(this=exp.Literal.number(min(50, policy.max_limit))))

    return parsed.sql(dialect="postgres")