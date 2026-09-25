"""Hudu (documentation) lookups for the Hercules chat assistant.

Read-only. Hercules can search a client's Hudu documentation — knowledge-base
articles, assets (applications, servers, network gear, and their fields) and the
company's own notes — the same way it searches other ConnectWise tickets.

Company mapping
---------------
Hudu's ConnectWise Manage integration stamps each Hudu company with the CW
company id (``integrations[].integrator_name == "cw_manage"``, ``sync_id`` =
CW company id), and ``GET /companies?id_in_integration=<cw id>`` filters on it.
That lookup is done ONCE per CW company and the answer is stored in Postgres
(``hudu_companies``) and in memory, so on a normal chat turn resolving the
company costs nothing and the only Hudu calls are the actual searches.

Passwords are deliberately never fetched: nothing here touches
``/asset_passwords`` and a search result never includes a password value.

The one write is ``create_article``: a knowledge-base article the assistant
drafted and the technician confirmed in the pod (see cw_tools / /action).

Configuration: ``HUDU_BASE_URL`` (e.g. https://acme.huducloud.com) and
``HUDU_API_KEY``. Without both, the Hudu tools are simply not offered.
"""

import asyncio
import html
import os
import re
import time
from html.parser import HTMLParser

import httpx

import db


class HuduError(Exception):
    pass


_client: httpx.AsyncClient | None = None
_base_url: str = ""

# CW company id -> {"hudu_company_id", "hudu_company_name", "hudu_url"} (or
# hudu_company_id None when Hudu has no such company). Warm for the process.
_company_cache: dict[int, dict] = {}
_company_cache_at: dict[int, float] = {}
_inflight: dict[int, asyncio.Task] = {}

# A "no match" answer is retried after this long — the company may get synced
# into Hudu later. A positive match is trusted until the pod restarts (and the
# DB row keeps it across restarts).
NO_MATCH_TTL_SECONDS = 6 * 3600


def init_client() -> None:
    global _client, _base_url
    base = (os.getenv("HUDU_BASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("HUDU_API_KEY") or "").strip()
    if not base or not key:
        print("[hudu] HUDU_BASE_URL / HUDU_API_KEY not set — Hudu lookups disabled")
        return
    _base_url = base
    _client = httpx.AsyncClient(
        base_url=f"{base}/api/v1",
        headers={"x-api-key": key, "Accept": "application/json"},
        timeout=20.0,
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


async def _get(path: str, params: dict | None = None):
    if _client is None:
        raise HuduError("Hudu is not configured")
    response = await _client.get(path, params={k: v for k, v in (params or {}).items() if v is not None})
    if response.status_code == 401:
        raise HuduError("Hudu authentication failed — check HUDU_API_KEY")
    if response.status_code == 404:
        raise HuduError("Not found in Hudu")
    if response.status_code >= 400:
        raise HuduError(f"Hudu API error {response.status_code}: {response.text[:160]}")
    return response.json()


def _rows(payload, key: str) -> list[dict]:
    """Hudu list endpoints answer {"articles": [...]}; tolerate a bare list too."""
    if isinstance(payload, dict):
        value = payload.get(key)
        return value if isinstance(value, list) else []
    return payload if isinstance(payload, list) else []


def _one(payload, key: str) -> dict | None:
    if isinstance(payload, dict):
        inner = payload.get(key)
        if isinstance(inner, dict):
            return inner
        if "id" in payload:
            return payload
    return None


def _absolute(url: str | None) -> str:
    if not url:
        return ""
    return url if url.startswith("http") else f"{_base_url}{url}"


# --- HTML -> text -------------------------------------------------------------


class _TextExtractor(HTMLParser):
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "blockquote", "pre", "table", "ul", "ol", "section"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")
        elif tag == "td" or tag == "th":
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(value, limit: int | None = None) -> str:
    """Flatten Hudu's rich-text HTML to readable plain text."""
    if value is None:
        return ""
    text = str(value)
    if "<" in text and ">" in text:
        parser = _TextExtractor()
        try:
            parser.feed(text)
            parser.close()
            text = "".join(parser.parts)
        except Exception:  # noqa: BLE001 - malformed HTML is still just text
            text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if limit and len(text) > limit:
        text = text[:limit].rstrip() + " …"
    return text


# --- Company mapping (CW company -> Hudu company) ------------------------------


def _shape_company(c: dict) -> dict:
    return {
        "hudu_company_id": c.get("id"),
        "hudu_company_name": c.get("name") or "",
        "hudu_url": _absolute(c.get("full_url") or c.get("url")),
    }


async def _lookup_company_in_hudu(cw_company_id: int, cw_company_name: str) -> dict:
    """The one real lookup: by ConnectWise sync id first, then by exact name."""
    rows = _rows(await _get("/companies", {"id_in_integration": cw_company_id, "page_size": 5}), "companies")
    match = next((c for c in rows if not c.get("archived")), None) or (rows[0] if rows else None)
    if match is None and cw_company_name:
        rows = _rows(await _get("/companies", {"name": cw_company_name, "page_size": 5}), "companies")
        wanted = cw_company_name.strip().lower()
        match = next((c for c in rows if (c.get("name") or "").strip().lower() == wanted), None)
    if match is None:
        return {"hudu_company_id": None, "hudu_company_name": "", "hudu_url": ""}
    return _shape_company(match)


async def resolve_company(cw_company_id: int | None, cw_company_name: str = "") -> dict | None:
    """Hudu company for a ConnectWise company, from memory -> Postgres -> Hudu.

    Returns None when Hudu has no matching company (or the id is missing).
    Concurrent callers for the same company share one lookup.
    """
    if not cw_company_id or not is_configured():
        return None
    cw_company_id = int(cw_company_id)

    cached = _company_cache.get(cw_company_id)
    if cached is not None:
        fresh = cached.get("hudu_company_id") or (time.time() - _company_cache_at.get(cw_company_id, 0) < NO_MATCH_TTL_SECONDS)
        if fresh:
            return cached if cached.get("hudu_company_id") else None

    task = _inflight.get(cw_company_id)
    if task is None:
        task = asyncio.create_task(_resolve_uncached(cw_company_id, cw_company_name))
        _inflight[cw_company_id] = task
        task.add_done_callback(lambda _t: _inflight.pop(cw_company_id, None))
    mapping = await task
    return mapping if mapping and mapping.get("hudu_company_id") else None


async def _resolve_uncached(cw_company_id: int, cw_company_name: str) -> dict:
    row = None
    try:
        row = await db.get_hudu_company(cw_company_id)
    except Exception as e:  # noqa: BLE001 - a DB blip just means one extra Hudu call
        print(f"[hudu] company cache read failed for CW {cw_company_id}: {e}")
    if row and (row.get("hudu_company_id") or row.get("age_seconds", 0) < NO_MATCH_TTL_SECONDS):
        mapping = {k: row.get(k) for k in ("hudu_company_id", "hudu_company_name", "hudu_url")}
    else:
        mapping = await _lookup_company_in_hudu(cw_company_id, cw_company_name)
        try:
            await db.save_hudu_company(cw_company_id, cw_company_name, mapping)
        except Exception as e:  # noqa: BLE001
            print(f"[hudu] company cache write failed for CW {cw_company_id}: {e}")
        print(f"[hudu] CW company {cw_company_id} ({cw_company_name!r}) -> Hudu "
              f"{mapping.get('hudu_company_id') or 'no match'}")
    _company_cache[cw_company_id] = mapping
    _company_cache_at[cw_company_id] = time.time()
    return mapping


def warm_company(cw_company_id: int | None, cw_company_name: str = "") -> None:
    """Fire-and-forget: resolve the mapping now so the first chat lookup is instant."""
    if not cw_company_id or not is_configured():
        return

    async def _warm():
        try:
            await resolve_company(cw_company_id, cw_company_name)
        except Exception as e:  # noqa: BLE001
            print(f"[hudu] warm-up failed for CW company {cw_company_id}: {e}")

    asyncio.create_task(_warm())


# --- Reads ---------------------------------------------------------------------


async def get_company(hudu_company_id: int) -> dict | None:
    payload = await _get(f"/companies/{int(hudu_company_id)}")
    c = _one(payload, "company")
    if not c:
        return None
    return {
        "id": c.get("id"),
        "name": c.get("name"),
        "nickname": c.get("nickname") or "",
        "company_type": c.get("company_type") or "",
        "url": _absolute(c.get("full_url") or c.get("url")),
        "notes": html_to_text(c.get("notes"), 2500),
    }


def _shape_article(a: dict, snippet: int = 320) -> dict:
    return {
        "article_id": a.get("id"),
        "title": a.get("name"),
        "company_id": a.get("company_id"),
        "url": _absolute(a.get("url")),
        "updated": (a.get("updated_at") or "")[:10],
        "snippet": html_to_text(a.get("content"), snippet),
    }


def _shape_asset(a: dict, field_chars: int = 160, max_fields: int = 8) -> dict:
    fields = []
    for f in a.get("fields") or []:
        value = html_to_text(f.get("value"), field_chars)
        if value:
            fields.append({"label": f.get("label"), "value": value})
    return {
        "asset_id": a.get("id"),
        "name": a.get("name"),
        "type": a.get("asset_type"),
        "company_id": a.get("company_id"),
        "url": _absolute(a.get("url")),
        "serial": a.get("primary_serial") or None,
        "model": a.get("primary_model") or None,
        "fields": fields[:max_fields],
    }


async def search_articles(query: str, company_id: int | None = None, limit: int = 8) -> list[dict]:
    params = {"search": query or None, "company_id": company_id, "page_size": limit}
    rows = _rows(await _get("/articles", params), "articles")
    return [_shape_article(a) for a in rows if not a.get("archived") and not a.get("draft")][:limit]


async def search_assets(query: str, company_id: int | None = None, limit: int = 8) -> list[dict]:
    params = {"search": query or None, "company_id": company_id, "page_size": limit}
    rows = _rows(await _get("/assets", params), "assets")
    return [_shape_asset(a) for a in rows if not a.get("archived")][:limit]


async def get_article(article_id: int, max_chars: int = 12_000) -> dict | None:
    a = _one(await _get(f"/articles/{int(article_id)}"), "article")
    if not a:
        return None
    shaped = _shape_article(a, snippet=0)
    shaped.pop("snippet", None)
    shaped["content"] = html_to_text(a.get("content"), max_chars)
    return shaped


async def get_asset(asset_id: int, max_chars: int = 12_000) -> dict | None:
    rows = _rows(await _get("/assets", {"id": int(asset_id), "page_size": 1}), "assets")
    if not rows:
        return None
    shaped = _shape_asset(rows[0], field_chars=max_chars, max_fields=60)
    shaped["company_name"] = rows[0].get("company_name")
    return shaped


# --- Write: knowledge-base article ---------------------------------------------

_STEP_RE = re.compile(r"^\s*(?:(\d+)[.)]|[-*\u2022])\s+(.*)$")


def text_to_html(text: str) -> str:
    """Plain-text article body -> the simple HTML the existing KB uses.

    Matches how NTM's short articles are written: a bold purpose line, then a
    numbered list of steps (``1. ...``) or bullets (``- ...``); other lines
    become paragraphs and blank lines separate blocks. No inline styling.
    """
    blocks: list[str] = []
    list_tag = None
    items: list[str] = []
    first_para = True

    def flush_list():
        nonlocal list_tag, items
        if list_tag and items:
            blocks.append(f"<{list_tag}>" + "".join(f"<li>{i}</li>" for i in items) + f"</{list_tag}>")
        list_tag, items = None, []

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            flush_list()
            continue
        m = _STEP_RE.match(line)
        if m:
            tag = "ol" if m.group(1) else "ul"
            if list_tag != tag:
                flush_list()
                list_tag = tag
            items.append(html.escape(m.group(2).strip()))
            continue
        flush_list()
        escaped = html.escape(line)
        lower = line.lower()
        if first_para or lower.startswith(("purpose:", "note:", "when to use:")):
            escaped = f"<strong>{escaped}</strong>"
        blocks.append(f"<p>{escaped}</p>")
        first_para = False
    flush_list()
    return "\n".join(blocks)


async def create_article(name: str, content_html: str, company_id: int | None = None,
                         folder_id: int | None = None) -> dict:
    """Create a KB article. company_id None = NTM-wide (global) knowledge base."""
    if _client is None:
        raise HuduError("Hudu is not configured")
    article = {"name": name.strip()[:200], "content": content_html}
    if company_id:
        article["company_id"] = int(company_id)
    if folder_id:
        article["folder_id"] = int(folder_id)
    response = await _client.post("/articles", json={"article": article})
    if response.status_code >= 400:
        raise HuduError(f"Hudu rejected the article ({response.status_code}): {response.text[:200]}")
    created = _one(response.json(), "article") or {}
    return {
        "id": created.get("id"),
        "name": created.get("name") or article["name"],
        "url": _absolute(created.get("url")),
        "company_id": created.get("company_id"),
    }
