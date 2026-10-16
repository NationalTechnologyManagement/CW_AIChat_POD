"""Client for ntm-sms-agent, the service that texts customers about tickets.

The pod's "Text" button goes through here: the browser never sees the agent's
URL or secret. The agent owns every texting rule (POC enrolment, STOP, cell-only,
test mode), so this module only relays its answer and status code.

Configured by SMS_AGENT_URL + SMS_AGENT_SECRET (the agent's TECH_API_SECRET).
A dedicated secret, never POD_SECRET or LIVE_BRIDGE_SECRET.
"""

import os

import httpx

_client: httpx.AsyncClient | None = None


def init_client() -> None:
    global _client
    base_url = os.getenv("SMS_AGENT_URL", "").strip().rstrip("/")
    secret = os.getenv("SMS_AGENT_SECRET", "").strip()
    if not base_url or not secret:
        _client = None
        return
    _client = httpx.AsyncClient(
        base_url=base_url,
        headers={"X-Tech-Secret": secret, "Accept": "application/json"},
        timeout=30.0,
    )


async def close_client() -> None:
    global _client
    if _client:
        await _client.aclose()
        _client = None


def is_configured() -> bool:
    return _client is not None


def _body(response: httpx.Response) -> dict:
    try:
        data = response.json()
    except ValueError:
        data = None
    if isinstance(data, dict):
        return data
    return {"success": False, "error": f"SMS agent answered {response.status_code}"}


async def ticket_info(ticket_id: int, member_identifier: str | None = None) -> tuple[int, dict]:
    """(status, body): who would be texted, whether it is allowed, recent texts,
    and the first name the text will be signed with for this member."""
    params = {"member": member_identifier} if member_identifier else None
    response = await _client.get(f"/api/tickets/{ticket_id}/sms", params=params)
    return response.status_code, _body(response)


async def send_text(ticket_id: int, message: str, member_identifier: str | None,
                    to: str | None = None) -> tuple[int, dict]:
    """`to` is a number the tech typed; the agent checks it is a cell and not opted out."""
    payload = {"message": message, "memberIdentifier": member_identifier}
    if to:
        payload["to"] = to
    response = await _client.post(f"/api/tickets/{ticket_id}/sms", json=payload)
    return response.status_code, _body(response)
