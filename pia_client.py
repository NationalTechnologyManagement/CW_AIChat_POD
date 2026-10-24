"""Pia (pia.ai) Tech Assist automations for the Hercules chat assistant.

Pia runs the service-desk automations NTM already offers techs through the Pia
pod in ConnectWise (unlock account, reset password, mailbox permissions, restart
services, ...). This module lets Hercules do the same from its own chat:

* ``readiness(cw_company_id)`` — is Pia set up for THIS ticket's client? Pia
  stores the ConnectWise company id on each of its clients (``psaCompanyId``),
  and Tech Assist automations are published per PSA company. A client that is
  not in Pia, or has no automations published to it, is "not ready" and the Pia
  tools are withheld from the model so it never offers what cannot run.
* ``techassist_options`` — the catalogue Pia's own pod shows: a button label,
  the "I need to ..." phrasing, a category and the package id to trigger.
* ``trigger`` — starts an automation in the context of a ConnectWise ticket.
  Pia then opens ITS OWN chat session on that ticket and asks for whatever it
  still needs (which user, which mailbox) as a form — the tech answers that in
  the Pia pod on the ticket. Hercules starts the right automation and tracks it;
  Pia gathers the inputs and does the work.
* ``execution_report`` — status + activity log for a started automation,
  reduced to one state (queued / waiting_for_input / running / completed /
  error / cancelled) and the question Pia is currently asking, if any.

Authentication is the API-app secret: ``Authorization: api <secret>``.
Configured by ``PIA_BASE_URL`` (https://<tenant>.pia.ai) and ``PIA_API_SECRET``
(vault secret ``pia-api-secret``). Without both, Pia is simply not offered.
"""

import asyncio
import os
import re
import time

import httpx


class PiaError(Exception):
    pass


_client: httpx.AsyncClient | None = None
_base_url: str = ""

# Pia has ~30 clients and the catalogue changes rarely; both are cached for
# the process so a chat turn costs no Pia calls. psaCompanyId is a string.
CACHE_TTL_SECONDS = int(os.getenv("PIA_CACHE_TTL_SECONDS", "600"))
_clients_cache: tuple[float, dict[str, dict]] | None = None
_catalog_cache: dict[int, tuple[float, list[dict]]] = {}
_readiness_cache: dict[int, tuple[float, dict]] = {}
_inflight: dict[str, asyncio.Task] = {}

MAX_PAGES = 10


def init_client() -> None:
    global _client, _base_url
    base = (os.getenv("PIA_BASE_URL") or "").strip().rstrip("/")
    secret = (os.getenv("PIA_API_SECRET") or "").strip()
    if not base or not secret:
        _client = None
        _base_url = ""
        return
    _base_url = base
    _client = httpx.AsyncClient(
        base_url=base,
        headers={"Authorization": f"api {secret}", "Accept": "application/json"},
        timeout=30.0,
    )


async def close_client() -> None:
    global _client
    if _client:
        await _client.aclose()
        _client = None


def is_configured() -> bool:
    return _client is not None


def base_url() -> str:
    return _base_url


# Pia's embeddable Tech Assist chat for one ticket — the page its own browser
# extension iframes into Autotask/Halo. Cookie (Entra) authenticated; the
# extension opens EXTENSION_AUTH_PATH in a popup when the cookie is missing.
EMBED_PATH = "/Extension/PiaEmbed/"
EXTENSION_AUTH_PATH = "/Extension/Authenticate"


def embed_url(ticket_id: int | None) -> str:
    if not _base_url or not ticket_id:
        return ""
    return f"{_base_url}{EMBED_PATH}?ticketId={int(ticket_id)}"


def auth_url() -> str:
    return f"{_base_url}{EXTENSION_AUTH_PATH}" if _base_url else ""


def _reset_caches() -> None:
    """Tests only."""
    global _clients_cache
    _clients_cache = None
    _catalog_cache.clear()
    _readiness_cache.clear()
    _inflight.clear()


# --- HTTP ----------------------------------------------------------------------


def _raise_for(response: httpx.Response, what: str) -> None:
    if response.status_code == 401 or response.status_code == 403:
        raise PiaError(f"Pia refused the API secret ({response.status_code}) while {what}")
    if response.status_code >= 400:
        detail = ""
        try:
            body = response.json()
            detail = body.get("detail") or body.get("title") or ""
        except ValueError:
            detail = response.text[:160]
        raise PiaError(f"Pia answered {response.status_code} while {what}: {detail}")


async def _get(path: str, params: dict | None = None):
    if _client is None:
        raise PiaError("Pia is not configured")
    response = await _client.get(path, params=params)
    _raise_for(response, f"reading {path}")
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


async def _get_all(path: str, params: dict | None = None) -> list[dict]:
    """Follow Pia's OData-style paging (``next`` is a relative URL)."""
    items: list[dict] = []
    payload = await _get(path, {**(params or {}), "$top": 100})
    for _ in range(MAX_PAGES):
        if not isinstance(payload, dict):
            break
        items.extend(payload.get("value") or [])
        nxt = payload.get("next")
        if not nxt or (payload.get("nextTop") == 0):
            break
        payload = await _get(nxt)
    return items


async def _shared(key: str, factory):
    """Coalesce concurrent identical lookups (pod load + first chat turn)."""
    task = _inflight.get(key)
    if task is None or task.done():
        task = asyncio.ensure_future(factory())
        _inflight[key] = task
    try:
        return await task
    finally:
        if _inflight.get(key) is task and task.done():
            _inflight.pop(key, None)


# --- Clients ---------------------------------------------------------------------


def _client_row(c: dict) -> dict:
    cid = c.get("id")
    return {
        "id": cid,
        "name": c.get("name") or "",
        "psa_company_id": str(c.get("psaCompanyId") or ""),
        "chatbot_enabled": bool(c.get("isPiaChatbotEnabled")),
        "tags": [t for t in (c.get("tags") or []) if isinstance(t, str)],
        "url": f"{_base_url}/r/clients/{cid}/dashboard" if cid else _base_url,
    }


async def _clients_by_psa() -> dict[str, dict]:
    global _clients_cache
    if _clients_cache and time.time() - _clients_cache[0] < CACHE_TTL_SECONDS:
        return _clients_cache[1]

    async def fetch():
        global _clients_cache
        rows = await _get_all("/api/config/clients")
        mapping = {}
        for c in rows:
            row = _client_row(c)
            if row["psa_company_id"]:
                mapping[row["psa_company_id"]] = row
        _clients_cache = (time.time(), mapping)
        return mapping

    return await _shared("clients", fetch)


async def resolve_client(cw_company_id: int | None) -> dict | None:
    """The Pia client whose psaCompanyId is this ConnectWise company, or None."""
    if not cw_company_id or not is_configured():
        return None
    return (await _clients_by_psa()).get(str(cw_company_id))


# --- Tech Assist catalogue ----------------------------------------------------------


def _option_row(o: dict) -> dict:
    return {
        "package_id": o.get("packageId") or "",
        "button": (o.get("optionButton") or "").strip(),
        "text": (o.get("optionText") or "").strip(),
        "category": (o.get("optionCategory") or "").strip(),
        "name": (o.get("automationName") or "").strip(),
        "description": (o.get("description") or "").strip(),
    }


async def techassist_options(cw_company_id: int | None) -> list[dict]:
    """Automations published to this PSA company, as Pia's pod lists them."""
    if not cw_company_id or not is_configured():
        return []
    cached = _catalog_cache.get(cw_company_id)
    if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    async def fetch():
        try:
            rows = await _get_all(f"/api/automate/discover/techassist/psacompany/{int(cw_company_id)}")
        except PiaError as exc:
            # Pia answers 500 for a company it has never heard of — that is
            # "nothing published", not an outage.
            if "500" in str(exc):
                rows = []
            else:
                raise
        options = [r for r in (_option_row(o) for o in rows) if r["package_id"]]
        options.sort(key=lambda r: (r["category"].lower(), r["button"].lower()))
        _catalog_cache[cw_company_id] = (time.time(), options)
        return options

    return await _shared(f"catalog:{cw_company_id}", fetch)


def cached_catalog(cw_company_id: int | None) -> list[dict]:
    cached = _catalog_cache.get(cw_company_id or 0)
    return cached[1] if cached else []


def find_option(cw_company_id: int | None, package_id: str) -> dict | None:
    pid = (package_id or "").strip().lower()
    for o in cached_catalog(cw_company_id):
        if o["package_id"].lower() == pid:
            return o
    return None


# --- Readiness --------------------------------------------------------------------------


NOT_CONFIGURED = {"configured": False, "enabled": False, "client": None, "automations": [], "reason": "not_configured"}


async def readiness(cw_company_id: int | None) -> dict:
    """Is Pia set up for this client? Checked BEFORE anything is offered.

    reason: "ready" | "not_in_pia" (no Pia client carries this CW company id) |
    "no_automations" (client exists but nothing is published to it) |
    "no_company" | "not_configured".
    """
    if not is_configured():
        return dict(NOT_CONFIGURED)
    if not cw_company_id:
        return {"configured": True, "enabled": False, "client": None, "automations": [], "reason": "no_company"}
    cached = _readiness_cache.get(cw_company_id)
    if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    client, options = await asyncio.gather(resolve_client(cw_company_id), techassist_options(cw_company_id))
    if client is None:
        reason = "not_in_pia"
    elif not options:
        reason = "no_automations"
    else:
        reason = "ready"
    result = {
        "configured": True,
        "enabled": reason == "ready",
        "client": client,
        "automations": options if client else [],
        "reason": reason,
    }
    _readiness_cache[cw_company_id] = (time.time(), result)
    return result


def cached_readiness(cw_company_id: int | None) -> dict | None:
    cached = _readiness_cache.get(cw_company_id or 0)
    return cached[1] if cached else None


def warm(cw_company_id: int | None) -> None:
    """Fire-and-forget: resolve readiness now so the first chat turn is instant."""
    if not cw_company_id or not is_configured():
        return

    async def _warm():
        try:
            await readiness(cw_company_id)
        except Exception as e:  # noqa: BLE001
            print(f"[pia] warm-up failed for CW company {cw_company_id}: {e}")

    asyncio.create_task(_warm())


def not_ready_message(ready: dict, company_name: str = "") -> str:
    who = company_name or "This client"
    reason = ready.get("reason")
    if reason == "not_in_pia":
        return f"{who} is not set up in Pia (no Pia client carries its ConnectWise company id)."
    if reason == "no_automations":
        return f"{who} is in Pia but has no Tech Assist automations published to it."
    if reason == "no_company":
        return "This ticket has no company, so Pia has nothing to run against."
    if reason == "not_configured":
        return "Pia is not configured on this service (PIA_BASE_URL / PIA_API_SECRET)."
    return ""


# --- Trigger + monitor -------------------------------------------------------------------


async def trigger(package_id: str, ticket_id: int) -> str:
    """Start an automation in the context of a ConnectWise ticket. Returns the
    event correlation id used to track it. No variable inputs are passed: the
    package keys are not discoverable through the API, and Pia asks the tech
    for anything it needs through its own chat on the ticket."""
    if _client is None:
        raise PiaError("Pia is not configured")
    pid = (package_id or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", pid):
        raise PiaError("That is not a Pia package id")
    response = await _client.post(
        f"/api/automate/trigger/{pid}/withTicket/{int(ticket_id)}",
        json={"variableInputs": []},
    )
    _raise_for(response, "starting the automation")
    data = response.json() if response.content else {}
    corr = (data or {}).get("eventCorrelationId") if isinstance(data, dict) else None
    if not corr:
        raise PiaError("Pia accepted the request but returned no correlation id")
    return str(corr)


async def event_status(correlation_id: str) -> dict | None:
    return await _get(f"/api/automate/monitor/event/{_corr(correlation_id)}/status")


async def event_logs(correlation_id: str) -> list[dict]:
    data = await _get(f"/api/automate/monitor/event/{_corr(correlation_id)}/activitylogs")
    return data if isinstance(data, list) else []


def _corr(value: str) -> str:
    v = (value or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", v):
        raise PiaError("That is not a Pia correlation id")
    return v


_FORM_PROMPT = re.compile(r"Message Text:\s*(.+)", re.S)


def describe(status: dict | None, logs: list[dict]) -> dict:
    """Reduce Pia's status + activity log to what a tech (or the model) needs."""
    execs = (status or {}).get("packageExecutions") or []
    ex = execs[-1] if execs else {}
    if not status:
        state = "unknown"
    elif not execs:
        state = "queued" if not status.get("isProcessed") else "unknown"
    elif ex.get("isError"):
        state = "error"
    elif ex.get("isCancelled"):
        state = "cancelled"
    elif ex.get("isCompleted"):
        state = "completed"
    elif ex.get("isWaiting"):
        state = "waiting_for_input"
    else:
        state = "running"

    prompt = ""
    waiting_on_form = False
    for entry in reversed(logs):
        msg = entry.get("logMessage") or ""
        if "Waiting for form submission" in msg:
            waiting_on_form = True
        m = _FORM_PROMPT.search(msg)
        if m and entry.get("activityStaticName") == "chat_interaction":
            prompt = " ".join(m.group(1).split())[:400]
            break
    if state == "waiting_for_input" and not waiting_on_form and not prompt:
        # Waiting on something other than the tech (an approval, an agent).
        pass

    errors = [e.get("logMessage", "")[:300] for e in logs if (e.get("logType") or "").lower() == "error"]
    recent = [
        f"{(e.get('logDate') or '')[11:19]} {e.get('activityStaticName') or ''}: {' '.join((e.get('logMessage') or '').split())[:200]}"
        for e in logs[-12:]
    ]

    summaries = {
        "queued": "Pia has queued the automation.",
        "running": "Pia is running the automation.",
        "waiting_for_input": "Pia is waiting for the technician to answer its question (the form is in Pia's chat "
                             "for this ticket — framed in the Hercules card, or the Pia pod)."
                             if (waiting_on_form or prompt) else "Pia is waiting (on an approval or an agent).",
        "completed": "The automation finished.",
        "error": "The automation failed.",
        "cancelled": "The automation was cancelled.",
        "unknown": "Pia has no record of that run yet.",
    }
    return {
        "state": state,
        "summary": summaries[state],
        "package_name": ex.get("packageName") or "",
        "package_instance_id": ex.get("packageInstanceId"),
        "started": ex.get("executionStart"),
        "ended": ex.get("executionEnd"),
        "pia_question": prompt,
        "errors": errors[-3:],
        "recent_log": recent,
    }


async def execution_report(correlation_id: str) -> dict:
    status, logs = await asyncio.gather(event_status(correlation_id), event_logs(correlation_id))
    report = describe(status, logs)
    report["correlation_id"] = _corr(correlation_id)
    return report
