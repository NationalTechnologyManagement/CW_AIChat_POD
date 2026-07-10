import asyncio
import os
import uuid

import httpx


EXTENSION_ID = "2d558935-686a-4bd0-9991-07539f5fe749"


class ScreenConnectError(Exception):
    pass


class ScreenConnectNotConfiguredError(ScreenConnectError):
    pass


_client: httpx.AsyncClient | None = None
_base_url = ""


def init_client() -> None:
    global _client, _base_url

    _base_url = os.getenv("SCREENCONNECT_BASE_URL", "").strip().rstrip("/")
    secret = os.getenv("SCREENCONNECT_API_SECRET", "").strip()
    origin = os.getenv("SCREENCONNECT_ORIGIN", "").strip()

    if not _base_url or not secret:
        _client = None
        return

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "CTRLAuthHeader": secret,
    }
    if origin:
        headers["Origin"] = origin

    _client = httpx.AsyncClient(
        headers=headers,
        timeout=15.0,
    )


async def close_client() -> None:
    global _client
    if _client:
        await _client.aclose()
        _client = None


def is_configured() -> bool:
    return _client is not None


def _service_path(method: str) -> str:
    extension_id = os.getenv("SCREENCONNECT_EXTENSION_ID", EXTENSION_ID).strip()
    return f"/App_Extensions/{extension_id}/Service.ashx/{method}"


def _read_field(value: dict, name: str):
    target = name.casefold()
    for key, field_value in value.items():
        if str(key).casefold() == target:
            return field_value
    return None


def _unwrap_sessions(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("d", "value", "sessions", "Sessions"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
    return []


async def get_sessions_by_name(computer_name: str) -> list[dict]:
    if not _client:
        raise ScreenConnectNotConfiguredError(
            "ScreenConnect is not configured on the Hercules server"
        )

    response = await _client.post(
        f"{_base_url}{_service_path('GetSessionsByName')}",
        json=[computer_name],
    )
    if response.status_code >= 400:
        detail = response.text[:300]
        raise ScreenConnectError(
            f"ScreenConnect returned HTTP {response.status_code}: {detail}"
        )

    try:
        raw_sessions = _unwrap_sessions(response.json())
    except ValueError as exc:
        raise ScreenConnectError("ScreenConnect returned invalid JSON") from exc

    exact_name = computer_name.casefold()
    sessions = []
    for raw in raw_sessions:
        session_id = str(_read_field(raw, "SessionID") or "").strip()
        name = str(_read_field(raw, "Name") or "").strip()
        session_type = _read_field(raw, "SessionType")
        if not session_id or name.casefold() != exact_name:
            continue
        if isinstance(session_type, str) and session_type.casefold() != "access":
            continue
        try:
            session_id = str(uuid.UUID(session_id))
        except ValueError:
            continue

        guest_count = _read_field(raw, "GuestConnectedCount")
        online = guest_count > 0 if isinstance(guest_count, (int, float)) else None
        sessions.append({
            "session_id": session_id,
            "name": name,
            "online": online,
        })
    return sessions


def build_launch_url(session_id: str, mode: str = "control") -> str:
    if not _base_url:
        raise ScreenConnectNotConfiguredError(
            "ScreenConnect is not configured on the Hercules server"
        )

    normalized_id = str(uuid.UUID(session_id))
    command = "JoinWithOptions" if mode == "backstage" else "Join"
    return f"{_base_url}/Host#Access///{normalized_id}/{command}"


async def resolve_computers(computer_names: list[str]) -> list[dict]:
    names = list(dict.fromkeys(name.strip() for name in computer_names if name.strip()))
    results = await asyncio.gather(*(get_sessions_by_name(name) for name in names))

    sessions = []
    seen = set()
    for requested_name, matches in zip(names, results):
        for match in matches:
            if match["session_id"] in seen:
                continue
            seen.add(match["session_id"])
            sessions.append({
                **match,
                "configuration_name": requested_name,
                "control_url": build_launch_url(match["session_id"], "control"),
                "backstage_url": build_launch_url(match["session_id"], "backstage"),
            })
    return sessions
