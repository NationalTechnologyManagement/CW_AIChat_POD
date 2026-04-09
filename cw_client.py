import os
import base64
from datetime import datetime, timedelta, timezone

import httpx


class CWAuthError(Exception):
    pass


class CWNotFoundError(Exception):
    pass


class CWAPIError(Exception):
    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"CW API error {status_code}: {detail}")


_client: httpx.AsyncClient | None = None


def _build_headers() -> dict:
    company_id = os.getenv("CW_AUTH_COMPANY_ID", os.getenv("CW_COMPANY_ID"))
    public_key = os.getenv("CW_PUBLIC_KEY")
    private_key = os.getenv("CW_PRIVATE_KEY")
    client_id = os.getenv("CW_CLIENT_ID")

    auth_string = f"{company_id}+{public_key}:{private_key}"
    encoded = base64.b64encode(auth_string.encode()).decode()

    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json",
        "clientId": client_id,
        "Accept": "application/vnd.connectwise.com+json; version=2022.1",
    }


def init_client():
    global _client
    base_url = os.getenv("CW_API_URL", "https://api-na.myconnectwise.net")
    entry_point = os.getenv("CW_ENTRY_POINT", "v4_6_release")
    _client = httpx.AsyncClient(
        base_url=f"{base_url}/{entry_point}/apis/3.0",
        headers=_build_headers(),
        timeout=30.0,
    )


async def close_client():
    global _client
    if _client:
        await _client.aclose()
        _client = None


def _handle_response(response: httpx.Response):
    if response.status_code == 401:
        raise CWAuthError("ConnectWise authentication failed — check API keys")
    if response.status_code == 404:
        raise CWNotFoundError("Resource not found in ConnectWise")
    if response.status_code >= 400:
        detail = response.text[:200]
        raise CWAPIError(response.status_code, detail)


async def get_ticket(ticket_id: int) -> dict:
    response = await _client.get(f"/service/tickets/{ticket_id}")
    _handle_response(response)
    data = response.json()

    return {
        "id": data.get("id"),
        "summary": data.get("summary", ""),
        "board": _nested_name(data, "board"),
        "status": _nested_name(data, "status"),
        "company_id": _nested_field(data, "company", "id"),
        "company_name": _nested_name(data, "company"),
        "company_identifier": _nested_field(data, "company", "identifier"),
        "contact_id": _nested_field(data, "contact", "id"),
        "contact_name": _nested_name(data, "contact"),
        "owner_identifier": _nested_field(data, "owner", "identifier"),
        "owner_name": _nested_name(data, "owner"),
        "resources": data.get("resources", ""),
        "type": _nested_name(data, "type"),
        "subtype": _nested_name(data, "subType"),
        "priority": _nested_name(data, "priority"),
        "initial_description": data.get("initialDescription", ""),
    }


async def get_ticket_notes(ticket_id: int) -> list[dict]:
    response = await _client.get(
        f"/service/tickets/{ticket_id}/notes",
        params={"pageSize": 50, "orderBy": "id desc"},
    )
    _handle_response(response)
    notes = response.json()

    return [
        {
            "id": n.get("id"),
            "text": n.get("text", ""),
            "internal": n.get("internalAnalysisFlag", False),
            "member": _nested_name(n, "member"),
            "date": n.get("dateCreated", ""),
        }
        for n in notes
        if n.get("text", "").strip()
    ]


async def search_tickets(conditions: str, page_size: int = 5) -> list[dict]:
    response = await _client.get(
        "/service/tickets",
        params={
            "conditions": conditions,
            "pageSize": page_size,
            "orderBy": "id desc",
        },
    )
    _handle_response(response)
    tickets = response.json()

    return [
        {
            "id": t.get("id"),
            "summary": t.get("summary", ""),
            "status": _nested_name(t, "status"),
            "company_name": _nested_name(t, "company"),
            "contact_name": _nested_name(t, "contact"),
            "date_entered": t.get("dateEntered", ""),
        }
        for t in tickets
    ]


async def create_ticket_note(
    ticket_id: int,
    text: str,
    member_identifier: str | None = None,
    resolution: bool = False,
) -> dict:
    payload = {
        "text": text,
        "internalAnalysisFlag": not resolution,
        "detailDescriptionFlag": False,
        "resolutionFlag": resolution,
    }
    if member_identifier:
        payload["member"] = {"identifier": member_identifier}

    response = await _client.post(
        f"/service/tickets/{ticket_id}/notes", json=payload
    )
    _handle_response(response)
    return response.json()


async def send_email_to_contact(
    ticket_id: int,
    text: str,
    member_identifier: str | None = None,
) -> dict:
    """Send email to ticket contact via a 0-hour time entry with Discussion notes."""
    now = datetime.now(timezone.utc)
    time_start = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "chargeToType": "ServiceTicket",
        "chargeToId": ticket_id,
        "timeStart": time_start,
        "timeEnd": time_start,
        "actualHours": 0,
        "billableOption": "DoNotBill",
        "notes": text,
        "addToDetailDescriptionFlag": True,
        "addToInternalAnalysisFlag": False,
        "addToResolutionFlag": False,
        "emailContactFlag": True,
        "emailResourceFlag": False,
        "emailCcFlag": False,
    }
    if member_identifier:
        payload["member"] = {"identifier": member_identifier}

    response = await _client.post("/time/entries", json=payload)
    _handle_response(response)
    return response.json()


async def create_time_entry(
    ticket_id: int,
    actual_hours: float,
    notes: str = "",
    member_identifier: str | None = None,
) -> dict:
    now = datetime.now(timezone.utc)
    time_start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_end = (now + timedelta(hours=actual_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "chargeToType": "ServiceTicket",
        "chargeToId": ticket_id,
        "actualHours": actual_hours,
        "timeStart": time_start,
        "timeEnd": time_end,
    }
    if notes:
        payload["notes"] = notes
    if member_identifier:
        payload["member"] = {"identifier": member_identifier}

    response = await _client.post("/time/entries", json=payload)
    _handle_response(response)
    return response.json()


def _nested_name(data: dict, key: str) -> str:
    obj = data.get(key)
    if isinstance(obj, dict):
        return obj.get("name", "")
    return ""


def _nested_field(data: dict, key: str, field: str):
    obj = data.get(key)
    if isinstance(obj, dict):
        return obj.get(field)
    return None
