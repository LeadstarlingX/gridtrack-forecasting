import re
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from app.config import settings

_bearer = HTTPBearer(auto_error=False)


def get_allowed_districts(
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> list[str] | None:
    """FastAPI dependency.

    Returns None  → GeneralObserver or dev mode (no SQL filter applied).
    Returns list  → Observer; list contains allowed H3 district IDs.
    Returns []    → Observer with zero sectors (all data blocked).

    When JWT_SECRET is not configured the service runs in open mode so that
    local development and the existing test suite require no token changes.
    """
    if not settings.jwt_secret:
        return None  # dev / open mode

    if creds is None:
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        payload = jwt.decode(
            creds.credentials,
            settings.jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},  # audience verified by the .NET gateway
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    role = payload.get("role", "")
    if not role or role == "GeneralObserver":
        return None  # no filter

    # Observer — extract districtId claims (.NET encodes them as a list or scalar)
    raw = payload.get("districtId", [])
    district_ids: list[str] = [raw] if isinstance(raw, str) else list(raw)
    return district_ids


def scope_pg_sql(sql: str, allowed_districts: list[str] | None) -> str:
    """Restrict a SELECT to the caller's districts.

    Strategy:
    - JOIN queries: outer-wrap works when the LLM includes d."DistrictId" in SELECT
      (all _PG_SCHEMA examples do this).
    - Single-table queries (COUNT, no JOIN): inject directly into the WHERE clause
      to avoid "column DistrictId does not exist" errors on aggregate results.

    allowed_districts=None  → no-op (GeneralObserver / dev mode)
    allowed_districts=[]    → block all rows (Observer with zero sectors)
    """
    if allowed_districts is None:
        return sql

    if not allowed_districts:
        return 'SELECT * FROM (SELECT NULL) __empty WHERE FALSE'

    safe = [re.sub(r"[^\w\-]", "", d) for d in allowed_districts if d]
    if not safe:
        return 'SELECT * FROM (SELECT NULL) __empty WHERE FALSE'

    csv = ", ".join(f"'{d}'" for d in safe)
    district_filter = f'"DistrictId" IN ({csv})'

    sql_upper = sql.upper()
    if re.search(r'\bJOIN\b', sql_upper):
        # JOIN query: outer wrap; DistrictId must be in SELECT (see _PG_SCHEMA examples).
        return f'SELECT * FROM ({sql}) __rbac WHERE {district_filter}'

    # Single-table query: inject into WHERE before any GROUP BY / ORDER BY / LIMIT
    # so the filter applies before aggregation and never hits an ambiguous column.
    for kw in ('GROUP BY', 'ORDER BY', 'LIMIT'):
        pos = sql_upper.find(kw)
        if pos == -1:
            continue
        has_where = sql_upper.rfind('WHERE', 0, pos) != -1
        glue = f' AND {district_filter} ' if has_where else f' WHERE {district_filter} '
        return sql[:pos] + glue + sql[pos:]

    # No GROUP BY / ORDER BY / LIMIT: append at end
    if 'WHERE' in sql_upper:
        return sql + f' AND {district_filter}'
    return sql + f' WHERE {district_filter}'