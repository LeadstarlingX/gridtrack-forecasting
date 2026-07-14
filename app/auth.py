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
    """Wrap a SELECT in a subquery restricted to the caller's districts.

    Wrapping (not regex injection) is safe for any SELECT shape:
    CTEs, UNIONs, subqueries, ORDER BY / GROUP BY / LIMIT all survive intact.

    allowed_districts=None  → no-op (GeneralObserver / dev mode)
    allowed_districts=[]    → block all rows (Observer with zero sectors)
    """
    if allowed_districts is None:
        return sql

    if not allowed_districts:
        # Structurally valid SQL that returns zero rows for any column set
        return 'SELECT * FROM (SELECT NULL) __empty WHERE FALSE'

    # Server-controlled values, but sanitise anyway: keep only word chars and hyphens
    safe = [re.sub(r"[^\w\-]", "", d) for d in allowed_districts if d]
    if not safe:
        return 'SELECT * FROM (SELECT NULL) __empty WHERE FALSE'

    csv = ", ".join(f"'{d}'" for d in safe)
    return f'SELECT * FROM ({sql}) __rbac WHERE "DistrictId" IN ({csv})'