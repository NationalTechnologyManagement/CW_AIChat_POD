import asyncio
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
        "board_id": _nested_field(data, "board", "id"),
        "status": _nested_name(data, "status"),
        "company_id": _nested_field(data, "company", "id"),
        "company_name": _nested_name(data, "company"),
        "company_identifier": _nested_field(data, "company", "identifier"),
        "contact_id": _nested_field(data, "contact", "id"),
        "contact_name": _nested_name(data, "contact"),
        "contact_email": data.get("contactEmailAddress", ""),
        "contact_phone": data.get("contactPhoneNumber", ""),
        "owner_identifier": _nested_field(data, "owner", "identifier"),
        "owner_name": _nested_name(data, "owner"),
        "resources": data.get("resources", ""),
        "team": _nested_name(data, "team"),
        "type": _nested_name(data, "type"),
        "type_id": _nested_field(data, "type", "id"),
        "subtype": _nested_name(data, "subType"),
        "subtype_id": _nested_field(data, "subType", "id"),
        "item": _nested_name(data, "item"),
        "item_id": _nested_field(data, "item", "id"),
        "priority": _nested_name(data, "priority"),
        "severity": data.get("severity", ""),
        "impact": data.get("impact", ""),
        "source": _nested_name(data, "source"),
        "site_name": data.get("siteName") or _nested_name(data, "site"),
        # dateEntered is not a top-level ticket field on GET — it lives in _info.
        "date_entered": data.get("dateEntered") or (data.get("_info") or {}).get("dateEntered", ""),
        "closed_date": data.get("closedDate", ""),
        "closed_by": data.get("closedBy", ""),
        "sla_status": data.get("slaStatus", ""),
        "date_responded": data.get("dateResponded", ""),
        "date_resolved": data.get("dateResolved", ""),
        "budget_hours": data.get("budgetHours"),
        "actual_hours": data.get("actualHours"),
        # POST-only in current CW REST — empty on GET; the original description
        # is the ticket's first Discussion note, which the full note feed carries.
        "initial_description": data.get("initialDescription", ""),
    }


# Pagination safety cap: 40 pages * 250 = 10,000 notes/entries per ticket. No
# real ticket comes close; the cap only guards against a runaway loop.
_MAX_PAGES = 40


async def _get_all_pages(path: str, params: dict, max_pages: int = _MAX_PAGES) -> list[dict]:
    """Fetch every page of a CW collection endpoint (CW caps pageSize at 1000)."""
    page_size = params.get("pageSize", 250)
    items: list[dict] = []
    page = 1
    prev_batch = None
    while page <= max_pages:
        response = await _client.get(path, params={**params, "page": page})
        _handle_response(response)
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            break
        # Guard against an endpoint ignoring the page param (would otherwise
        # collect _MAX_PAGES copies of page 1).
        if batch == prev_batch:
            break
        prev_batch = batch
        items.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return items


async def get_ticket_notes(ticket_id: int, limit: int | None = None) -> list[dict]:
    """Ticket notes, newest first. By default fetches EVERY note (all pages) so
    long tickets aren't silently truncated; pass limit for a cheap single page
    (e.g. enriching similar tickets)."""
    if limit:
        response = await _client.get(
            f"/service/tickets/{ticket_id}/notes",
            params={"pageSize": limit, "orderBy": "id desc"},
        )
        _handle_response(response)
        notes = response.json()
    else:
        notes = await _get_all_pages(
            f"/service/tickets/{ticket_id}/notes",
            {"pageSize": 250, "orderBy": "id desc"},
        )

    return [
        {
            "id": n.get("id"),
            "text": n.get("text", ""),
            "internal": n.get("internalAnalysisFlag", False),
            "detail": n.get("detailDescriptionFlag", False),
            "resolution": n.get("resolutionFlag", False),
            "member": _nested_name(n, "member") or _nested_name(n, "contact"),
            "date": n.get("dateCreated", ""),
        }
        for n in notes
        if n.get("text", "").strip()
    ]


async def get_ticket_time_entries(ticket_id: int) -> list[dict]:
    """Every time entry logged against a ticket, newest first — the real work
    log (what was done, by whom, for how long)."""
    entries = await _get_all_pages(
        "/time/entries",
        {
            "conditions": f'chargeToType="ServiceTicket" AND chargeToId={ticket_id}',
            "pageSize": 250,
            "orderBy": "timeStart desc",
        },
    )

    return [
        {
            "id": e.get("id"),
            "member": _nested_name(e, "member") or _nested_field(e, "member", "identifier")
                      or e.get("enteredBy") or "",
            "time_start": e.get("timeStart", ""),
            "time_end": e.get("timeEnd", ""),
            "hours": e.get("actualHours"),
            "billable": e.get("billableOption", ""),
            "notes": e.get("notes", "") or "",
            "internal_notes": e.get("internalNotes", "") or "",
            "email_sent": bool(e.get("emailContactFlag")),
        }
        for e in entries
    ]


async def get_ticket_configurations(ticket_id: int) -> list[dict]:
    """Return the ConnectWise configuration records attached to a ticket."""
    response = await _client.get(
        f"/service/tickets/{ticket_id}/configurations",
        params={"pageSize": 50},
    )
    _handle_response(response)
    references = response.json()

    async def normalize(item: dict) -> dict | None:
        configuration_id = item.get("id")
        name = (item.get("name") or item.get("identifier") or "").strip()

        # Manage often returns only an ID for a ticket configuration association.
        # Hydrate that reference before using its asset name as a computer name.
        if not name and configuration_id:
            detail_response = await _client.get(f"/company/configurations/{configuration_id}")
            _handle_response(detail_response)
            detail = detail_response.json()
            name = (detail.get("name") or detail.get("identifier") or "").strip()

        if not configuration_id or not name:
            return None
        return {"id": configuration_id, "name": name}

    configurations = await asyncio.gather(*(normalize(item) for item in references))
    return [item for item in configurations if item]


async def get_ticket_audit_trail(ticket_id: int) -> list[dict]:
    """The ticket's audit trail (status changes, emails sent, assignments,
    field edits — who did what, when), newest first."""
    # Old tickets can accrue enormous audit trails (every field edit / SLA
    # event); 8 pages = 2,000 entries is plenty and keeps the fetch bounded.
    entries = await _get_all_pages(
        "/system/audittrail",
        {"type": "Ticket", "id": ticket_id, "pageSize": 250},
        max_pages=8,
    )

    normalized = [
        {
            "text": e.get("text", "") or "",
            "member": e.get("enteredBy", "") or "",
            "date": e.get("enteredDate", "") or "",
            "type": e.get("auditType", "") or e.get("auditSubType", "") or "",
        }
        for e in entries
        if (e.get("text") or "").strip()
    ]
    # This endpoint has no orderBy and its ordering is undocumented — sort
    # newest first ourselves (ISO dates sort lexically) so downstream trimming
    # keeps the recent end.
    normalized.sort(key=lambda a: a["date"], reverse=True)
    return normalized


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


def quote_literal(value) -> str:
    """A quoted string literal for a ConnectWise conditions clause.

    ConnectWise has NO escape sequence inside a conditions literal — doubling a
    quote does not escape it, it ends the string and 400s the whole query. What
    works is picking the delimiter the value doesn't contain, so a search for
    "O'Brien" or 'say "hi"' goes through instead of hard-failing the lookup.
    """
    text = str(value)
    if '"' not in text:
        return f'"{text}"'
    if "'" not in text:
        return f"'{text}'"
    return '"' + text.replace('"', "") + '"'


async def find_tickets(
    keywords: list[str] | None = None,
    match: str = "any",
    company_id: int | None = None,
    company_name: str = "",
    contact_id: int | None = None,
    contact_name: str = "",
    status_name: str = "",
    board_name: str = "",
    include_closed: bool = True,
    days_back: int | None = 365,
    exclude_ticket_id: int | None = None,
    limit: int = 8,
) -> list[dict]:
    """Structured ticket search across the WHOLE ConnectWise instance.

    Every filter is built here from typed parameters — a caller (including the
    AI assistant's search tool) never supplies a raw conditions string, so it
    cannot reach fields it wasn't given or break out of the quoting.
    """
    clauses: list[str] = []

    terms = [k.strip() for k in (keywords or []) if k and k.strip()]
    if terms:
        joiner = " and " if match == "all" else " or "
        clauses.append("(" + joiner.join(f"summary contains {quote_literal(t)}" for t in terms) + ")")
    if company_id:
        clauses.append(f"company/id = {int(company_id)}")
    elif company_name:
        clauses.append(f"company/name contains {quote_literal(company_name)}")
    if contact_id:
        clauses.append(f"contact/id = {int(contact_id)}")
    elif contact_name:
        clauses.append(f"contact/name contains {quote_literal(contact_name)}")
    if status_name:
        # Equality only: `status/name contains` is not a verified operator on
        # this field, and a rejected condition fails the entire search.
        clauses.append(f"status/name = {quote_literal(status_name)}")
    if board_name:
        clauses.append(f"board/name contains {quote_literal(board_name)}")
    if not include_closed:
        clauses.append("closedFlag = false")
    if days_back:
        since = datetime.now(timezone.utc) - timedelta(days=int(days_back))
        clauses.append(f"dateEntered > [{since.strftime('%Y-%m-%dT00:00:00Z')}]")
    if exclude_ticket_id:
        clauses.append(f"id != {int(exclude_ticket_id)}")

    # A date window alone is not a search — without something selective this
    # would just hand back the newest tickets in the instance.
    selective = bool(terms or company_id or company_name or contact_id
                     or contact_name or status_name or board_name)
    if not selective:
        return []

    response = await _client.get(
        "/service/tickets",
        params={
            "conditions": " and ".join(clauses),
            "pageSize": max(1, min(int(limit), 25)),
            "orderBy": "id desc",
        },
    )
    _handle_response(response)

    return [
        {
            "id": t.get("id"),
            "summary": t.get("summary", ""),
            "status": _nested_name(t, "status"),
            "board": _nested_name(t, "board"),
            "company_name": _nested_name(t, "company"),
            "contact_name": _nested_name(t, "contact"),
            "priority": _nested_name(t, "priority"),
            "closed": bool(t.get("closedFlag")),
            "date_entered": t.get("dateEntered") or (t.get("_info") or {}).get("dateEntered", ""),
        }
        for t in response.json()
    ]


async def search_contact_tickets(contact_id: int, exclude_ticket_id: int, since_date: str, keyword_conditions: str = "", page_size: int = 5) -> list[dict]:
    """Search tickets for a specific contact, optionally filtered by keywords."""
    conditions = f"contact/id={contact_id} and id != {exclude_ticket_id} and dateEntered > [{since_date}]"
    if keyword_conditions:
        conditions += f" and ({keyword_conditions})"
    return await search_tickets(conditions, page_size)


async def search_company_tickets(company_id: int, exclude_ticket_id: int, since_date: str, keyword_conditions: str = "", page_size: int = 5) -> list[dict]:
    """Search tickets for a specific company, optionally filtered by keywords."""
    conditions = f"company/id={company_id} and id != {exclude_ticket_id} and dateEntered > [{since_date}]"
    if keyword_conditions:
        conditions += f" and ({keyword_conditions})"
    return await search_tickets(conditions, page_size)


async def search_all_tickets(exclude_ticket_id: int, since_date: str, keyword_conditions: str, page_size: int = 5) -> list[dict]:
    """Search all tickets filtered by keywords."""
    conditions = f"id != {exclude_ticket_id} and dateEntered > [{since_date}] and ({keyword_conditions})"
    return await search_tickets(conditions, page_size)


async def create_ticket_note(
    ticket_id: int,
    text: str,
    member_identifier: str | None = None,
    internal: bool = True,
    resolution: bool = False,
    detail: bool = False,
) -> dict:
    payload = {
        "text": text,
        "internalAnalysisFlag": internal,
        "detailDescriptionFlag": detail,
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
    resolution: bool = False,
) -> dict:
    """Email the ticket contact via a 0-hour time entry.

    By default the text is posted as a Discussion note (a mid-ticket customer
    update). When resolution=True it is recorded as the ticket's Resolution — the
    customer-facing summary of how the issue was fixed — instead of Discussion, so
    the emailed message and the Resolution field are the same thing. (The internal
    technician analysis is written separately and stays on Internal Analysis.)"""
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
        "addToDetailDescriptionFlag": not resolution,
        "addToInternalAnalysisFlag": False,
        "addToResolutionFlag": resolution,
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
    time_start: str,
    time_end: str,
    notes: str = "",
    member_identifier: str | None = None,
    add_to_internal: bool = False,
    add_to_resolution: bool = False,
) -> dict:
    """Create a time entry from explicit start/end timestamps.

    time_start / time_end are ISO-8601 UTC strings (e.g. 2025-01-15T09:00:00Z).
    actualHours is derived from the span so the entry reflects the real worked
    time. The entry is attributed to member_identifier — pass the tech who
    actually did the work so it is never logged against the API/automation user.

    When add_to_internal is set, the notes are also posted to the ticket's
    Internal Analysis tab, so a single entry covers both time and internal notes.
    """
    start_dt = _parse_iso(time_start)
    end_dt = _parse_iso(time_end)
    actual_hours = round((end_dt - start_dt).total_seconds() / 3600, 2)

    payload = {
        "chargeToType": "ServiceTicket",
        "chargeToId": ticket_id,
        "actualHours": actual_hours,
        # ConnectWise rejects fractional seconds ("UnsupportedFormat"); the
        # browser's toISOString() includes milliseconds, so normalize to
        # whole-second UTC (yyyy-MM-ddTHH:mm:ssZ) here.
        "timeStart": _cw_datetime(start_dt),
        "timeEnd": _cw_datetime(end_dt),
    }
    if notes:
        payload["notes"] = notes
        payload["addToInternalAnalysisFlag"] = add_to_internal
        payload["addToResolutionFlag"] = add_to_resolution
    if member_identifier:
        payload["member"] = {"identifier": member_identifier}

    response = await _client.post("/time/entries", json=payload)
    _handle_response(response)
    return response.json()


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp (accepting a trailing 'Z') to a datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _cw_datetime(dt: datetime) -> str:
    """Format a datetime the way ConnectWise accepts: whole-second UTC with a
    trailing Z, no fractional seconds (which trigger 'UnsupportedFormat')."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def get_members() -> list[dict]:
    """Active full-license members (real technicians) for the time-entry picker.

    Filters out API/integration accounts (licenseClass != 'F') so the tech
    selecting themselves only sees real people.
    """
    response = await _client.get(
        "/system/members",
        params={
            "conditions": "inactiveFlag = false and licenseClass = 'F'",
            "fields": "id,identifier,firstName,lastName",
            "pageSize": 1000,
            "orderBy": "firstName asc",
        },
    )
    _handle_response(response)
    members = response.json()

    result = []
    for m in members:
        identifier = m.get("identifier")
        if not identifier:
            continue
        name = " ".join(p for p in [m.get("firstName"), m.get("lastName")] if p).strip()
        result.append({"identifier": identifier, "name": name or identifier})
    return result


async def get_board_statuses(board_id: int) -> list[dict]:
    """Statuses defined on a service board (used to resolve a status by name)."""
    response = await _client.get(
        f"/service/boards/{board_id}/statuses",
        params={"pageSize": 200, "orderBy": "sortOrder asc"},
    )
    _handle_response(response)
    return [
        {
            "id": s.get("id"),
            "name": s.get("name", ""),
            # The board-status endpoint returns "inactive"; "inactiveFlag" is the
            # spelling on other CW records. Read both so retired statuses are
            # actually filtered out instead of silently offered to a technician.
            "inactive": s.get("inactive", s.get("inactiveFlag", False)),
            "closed": s.get("closedStatus", False),
            # Some statuses (Re-Opened, the DNU placeholders) reject time entries
            # outright — worth knowing before a status change is proposed.
            "no_time_entry": s.get("timeEntryNotAllowed", False),
        }
        for s in response.json()
    ]


def _status_key(name: str) -> str:
    """Normalize a status name for matching: lowercase, letters/digits only."""
    return "".join(c for c in (name or "").lower() if c.isalnum())


def usable_statuses(statuses: list[dict]) -> list[dict]:
    """Active statuses a technician would actually pick.

    Real boards carry retired scaffolding — "DNU-*", "AUTOMATED STATUS BELOW (DO
    NOT USE)", "OLD STATUSES (DO NOT USE)", *-Automation. Support Tier 1 (board 49)
    instead marks its automation-driven statuses with a trailing "*" ("Client
    Contact 1*", "Closed Pending*"). None of those should ever reach a status
    picker or the assistant's list of options.
    """
    keep = []
    for s in statuses:
        name = (s.get("name") or "").strip().lower()
        if s.get("inactive") or not name:
            continue
        if "automat" in name or "do not use" in name or name.startswith("dnu") or name.endswith("*"):
            continue
        keep.append(s)
    return keep


def match_status(statuses: list[dict], wanted: str) -> dict | None:
    """Best status on a board for a spoken name ("in progress", "waiting on client").

    Exact normalized match first, then prefix, then substring either way — so
    "in progress" finds "In Progress" and "Working Issue Now > In Progress"
    without ever guessing between two equally-good candidates.
    """
    target = _status_key(wanted)
    if not target:
        return None
    active = usable_statuses(statuses)
    for test in (
        lambda k: k == target,
        lambda k: k.startswith(target),
        lambda k: target in k,
        lambda k: k in target,
    ):
        hits = [s for s in active if test(_status_key(s.get("name", "")))]
        if hits:
            # Shortest name = the least-qualified status matching the phrase.
            return min(hits, key=lambda s: len(s.get("name") or ""))
    return None


async def set_ticket_status(ticket_id: int, status_id: int) -> dict:
    """Move a ticket to a new status via a JSON-Patch replace."""
    payload = [{"op": "replace", "path": "status/id", "value": status_id}]
    response = await _client.patch(f"/service/tickets/{ticket_id}", json=payload)
    _handle_response(response)
    return response.json()


async def get_board_type_associations(board_id: int) -> list[dict]:
    """Valid Type/Subtype/Item combinations for a board.

    ConnectWise validates a ticket's categorization against these, and requires
    it before a ticket can be resolved/closed. Returns flat combos; the caller
    builds the dependent picker. Paginates so large boards aren't truncated.
    """
    combos: list[dict] = []
    page = 1
    while page <= 6:  # safety cap (6 * 1000 = 6000 combos)
        response = await _client.get(
            f"/service/boards/{board_id}/typeSubTypeItemAssociations",
            params={"pageSize": 1000, "page": page},
        )
        _handle_response(response)
        batch = response.json()
        if not batch:
            break
        for a in batch:
            combos.append({
                "type": {"id": _nested_field(a, "type", "id"), "name": _nested_name(a, "type")},
                "subtype": {"id": _nested_field(a, "subType", "id"), "name": _nested_name(a, "subType")},
                "item": {"id": _nested_field(a, "item", "id"), "name": _nested_name(a, "item")},
            })
        if len(batch) < 1000:
            break
        page += 1
    return combos


async def update_ticket_category(
    ticket_id: int,
    type_id: int | None = None,
    subtype_id: int | None = None,
    item_id: int | None = None,
) -> dict | None:
    """Set a ticket's Type / Subtype / Item via JSON-Patch (only the parts given)."""
    ops = []
    if type_id:
        ops.append({"op": "replace", "path": "type/id", "value": type_id})
    if subtype_id:
        ops.append({"op": "replace", "path": "subType/id", "value": subtype_id})
    if item_id:
        ops.append({"op": "replace", "path": "item/id", "value": item_id})
    if not ops:
        return None
    response = await _client.patch(f"/service/tickets/{ticket_id}", json=ops)
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
