import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from functools import partial

import psycopg


def _parse_ts(ts):
    """Parse an ISO-8601 send-time into an aware datetime, or None if unusable."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None

_conninfo: str | None = None
_use_sync: bool = False

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id          SERIAL PRIMARY KEY,
    ticket_id   INTEGER NOT NULL,
    role        VARCHAR(20) NOT NULL,
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_ticket ON chat_messages (ticket_id, id);

CREATE TABLE IF NOT EXISTS board_options (
    board_id    INTEGER PRIMARY KEY,
    data        TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Live messaging: a real-time technician<->customer chat, kept SEPARATE from
-- chat_messages so the AI-chat "Clear" button can never wipe a real customer
-- conversation. id is supplied by the publisher (UUID) so persistence is
-- idempotent across multiple subscribers/replicas (ON CONFLICT DO NOTHING).
CREATE TABLE IF NOT EXISTS live_messages (
    id                UUID PRIMARY KEY,
    ticket_id         INTEGER NOT NULL,
    sender            VARCHAR(16) NOT NULL,   -- 'technician' | 'customer' | 'system'
    member_identifier TEXT,
    author_name       TEXT,
    body              TEXT NOT NULL,
    attachments       TEXT,                   -- JSON array of {name,isImage,dataUrl}; NULL when none
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_live_messages_ticket ON live_messages (ticket_id, created_at, id);
-- Backfill on instances whose live_messages table predates the attachments column.
ALTER TABLE live_messages ADD COLUMN IF NOT EXISTS attachments TEXT;

CREATE TABLE IF NOT EXISTS live_sessions (
    ticket_id   INTEGER PRIMARY KEY,
    status      VARCHAR(16) NOT NULL DEFAULT 'active',  -- 'active' | 'ended'
    started_by  TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at    TIMESTAMPTZ
);

-- Phone calls: one row per Twilio parent Call SID. Written in pieces as the
-- call progresses (the transcript starts flowing before anyone has created a
-- ticket), so every column but call_key is nullable and upsert_call_session
-- only writes the fields it was actually given.
CREATE TABLE IF NOT EXISTS call_sessions (
    call_key        TEXT PRIMARY KEY,       -- Twilio parent Call SID
    ticket_id       INTEGER,                -- NULL until the call is tied to a ticket
    caller_number   TEXT,
    contact_id      INTEGER,
    company_id      INTEGER,
    tech_identifier TEXT,                   -- ConnectWise member identifier of who answered
    tech_name       TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    talk_seconds    INTEGER,
    disposition     TEXT,
    summary         TEXT,                   -- generated technician notes, filled at call end
    summary_state   TEXT DEFAULT 'pending'  -- 'pending' | 'ready' | 'failed' | 'posted'
);
CREATE INDEX IF NOT EXISTS idx_call_sessions_ticket ON call_sessions (ticket_id);
-- Retention sweep (purge_old_calls) scans by age; without this it seq-scans.
CREATE INDEX IF NOT EXISTS idx_call_sessions_started ON call_sessions (started_at);

-- Transcript segments. id is supplied by the sender (UUID) so a retried or
-- replayed batch is harmless (ON CONFLICT DO NOTHING), and (call_key, seq) is
-- unique so the same utterance can't land twice under two different ids.
-- ON DELETE CASCADE is what makes the 60-day purge a single DELETE.
CREATE TABLE IF NOT EXISTS call_transcript_segments (
    id          UUID PRIMARY KEY,
    call_key    TEXT NOT NULL REFERENCES call_sessions (call_key) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,       -- monotonic per call, for ordering
    speaker     VARCHAR(16) NOT NULL,   -- 'customer' | 'technician'
    text        TEXT NOT NULL,
    spoken_at   TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (call_key, seq)
);
CREATE INDEX IF NOT EXISTS idx_call_segments_call ON call_transcript_segments (call_key, seq);

-- ConnectWise company -> Hudu company, resolved once and kept. hudu_company_id
-- is NULL when Hudu had no match at resolved_at (retried after a while, see
-- hudu_client.NO_MATCH_TTL_SECONDS).
CREATE TABLE IF NOT EXISTS hudu_companies (
    cw_company_id     INTEGER PRIMARY KEY,
    cw_company_name   TEXT,
    hudu_company_id   INTEGER,
    hudu_company_name TEXT,
    hudu_url          TEXT,
    resolved_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Admin dashboard settings, SHARED with the customer-facing Hercules service
-- (which serves the dashboard at /admin). app = 'client' | 'internal'. Rows keyed
-- 'default.<key>' are each app's built-in values, published at boot; a plain
-- '<key>' row is an admin override. Either service may create the tables.
CREATE TABLE IF NOT EXISTS hercules_admin_settings (
    app        TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by TEXT,
    PRIMARY KEY (app, key)
);
CREATE TABLE IF NOT EXISTS hercules_admin_settings_history (
    id         BIGSERIAL PRIMARY KEY,
    app        TEXT NOT NULL,
    key        TEXT NOT NULL,
    old_value  TEXT,
    new_value  TEXT,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    changed_by TEXT
);
CREATE INDEX IF NOT EXISTS hercules_admin_settings_history_app_idx
    ON hercules_admin_settings_history (app, changed_at DESC);
"""


def _sync_query(conninfo: str, query: str, params: tuple = ()):
    """Run a query synchronously and return all rows."""
    with psycopg.Connection.connect(conninfo) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall()


def _sync_execute(conninfo: str, query: str, params: tuple = ()):
    """Run a write query synchronously and return rowcount."""
    with psycopg.Connection.connect(conninfo) as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.rowcount


def _sync_execute_many(conninfo: str, query: str, params_seq: list):
    """Run one write query over many parameter sets on a SINGLE connection and
    return the total rowcount.

    Everything else in this module opens a connection per query, which is fine
    at chat speed. Transcript segments arrive in bursts, so the per-query shape
    would mean one Postgres connection per spoken utterance."""
    with psycopg.Connection.connect(conninfo) as conn:
        cur = conn.cursor()
        cur.executemany(query, params_seq)
        conn.commit()
        return cur.rowcount


async def init_pool():
    global _conninfo, _use_sync
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[db] DATABASE_URL not set — chat persistence disabled")
        return

    _conninfo = database_url

    try:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            await conn.execute(SCHEMA_SQL)
        print("[db] Connected and schema ready")
    except psycopg.InterfaceError as e:
        if "ProactorEventLoop" in str(e) and sys.platform == "win32":
            _use_sync = True
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, partial(_sync_execute, _conninfo, SCHEMA_SQL))
            print("[db] Connected (sync fallback for Windows) and schema ready")
        else:
            raise


async def close_pool():
    global _conninfo
    _conninfo = None


async def get_messages(ticket_id: int) -> list[dict]:
    if not _conninfo:
        return []

    query = "SELECT role, content FROM chat_messages WHERE ticket_id = %s ORDER BY id ASC"

    if _use_sync:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, (ticket_id,)))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, (ticket_id,))
            rows = await cur.fetchall()

    return [{"role": r[0], "content": _parse_content(r[1])} for r in rows]


def _parse_content(raw: str):
    """Restore structured (multimodal) content stored as JSON; plain text passes through."""
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(p, dict) and "type" in p for p in parsed):
                return parsed
        except json.JSONDecodeError:
            pass
    return raw


async def save_message(ticket_id: int, role: str, content) -> int | None:
    if not _conninfo:
        return None

    if not isinstance(content, str):
        content = json.dumps(content)

    query = "INSERT INTO chat_messages (ticket_id, role, content) VALUES (%s, %s, %s) RETURNING id"

    if _use_sync:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, (ticket_id, role, content)))
        return rows[0][0] if rows else None
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, (ticket_id, role, content))
            result = await cur.fetchone()
            return result[0] if result else None


async def get_board_options(board_id: int):
    """Return (options_dict, age_seconds) for a board's cached type/subtype/item
    tree, or None if not cached. Age lets the caller decide if it's stale."""
    if not _conninfo:
        return None

    query = (
        "SELECT data, EXTRACT(EPOCH FROM (NOW() - updated_at)) "
        "FROM board_options WHERE board_id = %s"
    )

    if _use_sync:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, (board_id,)))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, (board_id,))
            rows = await cur.fetchall()

    if not rows:
        return None
    try:
        return json.loads(rows[0][0]), float(rows[0][1])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


async def save_board_options(board_id: int, data) -> None:
    if not _conninfo:
        return

    payload = json.dumps(data)
    query = (
        "INSERT INTO board_options (board_id, data, updated_at) VALUES (%s, %s, NOW()) "
        "ON CONFLICT (board_id) DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()"
    )

    if _use_sync:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(_sync_execute, _conninfo, query, (board_id, payload)))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            await conn.execute(query, (board_id, payload))


async def clear_messages(ticket_id: int) -> int:
    if not _conninfo:
        return 0

    query = "DELETE FROM chat_messages WHERE ticket_id = %s"

    if _use_sync:
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, partial(_sync_execute, _conninfo, query, (ticket_id,)))
        return count
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, (ticket_id,))
            return cur.rowcount


# --- Live messaging ---


async def save_live_message(
    msg_id: str,
    ticket_id: int,
    sender: str,
    body: str,
    member_identifier: str | None = None,
    author_name: str | None = None,
    ts: str | None = None,
    attachments=None,
) -> bool:
    """Persist one live message. Idempotent: a duplicate id (same message seen by
    another subscriber/replica) is silently ignored. Returns True if a row was
    actually inserted (i.e. this was the first time we saw this id).

    created_at is stored from the envelope's send-time `ts` so the transcript
    orders by when messages were sent, not when this replica happened to persist
    them (which can differ under network jitter / multiple subscribers).

    `attachments` (a list of {name,isImage,dataUrl}) is stored as a JSON string,
    same convention as chat_messages' structured content."""
    if not _conninfo:
        return False

    created_at = _parse_ts(ts) or datetime.now(timezone.utc)
    attachments_json = json.dumps(attachments) if attachments else None
    query = (
        "INSERT INTO live_messages (id, ticket_id, sender, member_identifier, author_name, body, attachments, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING"
    )
    params = (msg_id, ticket_id, sender, member_identifier, author_name, body, attachments_json, created_at)

    if _use_sync:
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, partial(_sync_execute, _conninfo, query, params))
        return bool(count)
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, params)
            return bool(cur.rowcount)


async def get_live_messages(ticket_id: int) -> list[dict]:
    """Full live-chat transcript for a ticket, oldest first (for backlog replay
    and the end-of-chat summary)."""
    if not _conninfo:
        return []

    query = (
        "SELECT id, sender, member_identifier, author_name, body, attachments, created_at "
        "FROM live_messages WHERE ticket_id = %s ORDER BY created_at ASC, id ASC"
    )

    if _use_sync:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, (ticket_id,)))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, (ticket_id,))
            rows = await cur.fetchall()

    def _atts(raw):
        if not raw:
            return []
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except (ValueError, TypeError):
            return []

    return [
        {
            "id": str(r[0]),
            "sender": r[1],
            "memberIdentifier": r[2],
            "authorName": r[3],
            "body": r[4],
            "attachments": _atts(r[5]),
            "ts": r[6].isoformat() if r[6] else None,
        }
        for r in rows
    ]


async def start_live_session(ticket_id: int, started_by: str | None = None) -> None:
    if not _conninfo:
        return

    query = (
        "INSERT INTO live_sessions (ticket_id, status, started_by, started_at, ended_at) "
        "VALUES (%s, 'active', %s, NOW(), NULL) "
        "ON CONFLICT (ticket_id) DO UPDATE SET "
        "status = 'active', started_by = EXCLUDED.started_by, started_at = NOW(), ended_at = NULL"
    )
    params = (ticket_id, started_by)

    if _use_sync:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(_sync_execute, _conninfo, query, params))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            await conn.execute(query, params)


async def end_live_session(ticket_id: int) -> None:
    if not _conninfo:
        return

    query = "UPDATE live_sessions SET status = 'ended', ended_at = NOW() WHERE ticket_id = %s"

    if _use_sync:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(_sync_execute, _conninfo, query, (ticket_id,)))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            await conn.execute(query, (ticket_id,))


async def get_live_session(ticket_id: int) -> dict | None:
    if not _conninfo:
        return None

    query = (
        "SELECT status, started_by, started_at, ended_at "
        "FROM live_sessions WHERE ticket_id = %s"
    )

    if _use_sync:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, (ticket_id,)))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, (ticket_id,))
            rows = await cur.fetchall()

    if not rows:
        return None
    r = rows[0]
    return {
        "status": r[0],
        "started_by": r[1],
        "started_at": r[2].isoformat() if r[2] else None,
        "ended_at": r[3].isoformat() if r[3] else None,
    }


# --- Call transcripts ---

# Columns upsert_call_session is allowed to write, in a fixed order. Anything
# else the sender puts in the payload is ignored rather than raising: the caller
# is a separate service and a stray field must not fail an ingest.
_CALL_SESSION_FIELDS = (
    "ticket_id",
    "caller_number",
    "contact_id",
    "company_id",
    "tech_identifier",
    "tech_name",
    "started_at",
    "ended_at",
    "talk_seconds",
    "disposition",
    "summary",
    "summary_state",
)

_CALL_SESSION_INT_FIELDS = ("ticket_id", "contact_id", "company_id", "talk_seconds")
_CALL_SESSION_TS_FIELDS = ("started_at", "ended_at")

_CALL_SESSION_SELECT = (
    "SELECT call_key, ticket_id, caller_number, contact_id, company_id, tech_identifier, "
    "tech_name, started_at, ended_at, talk_seconds, disposition, summary, summary_state "
    "FROM call_sessions "
)


def _call_session_row(r) -> dict:
    """Map a _CALL_SESSION_SELECT row to the dict shape the pod/API speak."""
    return {
        "call_key": r[0],
        "ticket_id": r[1],
        "caller_number": r[2],
        "contact_id": r[3],
        "company_id": r[4],
        "tech_identifier": r[5],
        "tech_name": r[6],
        "started_at": r[7].isoformat() if r[7] else None,
        "ended_at": r[8].isoformat() if r[8] else None,
        "talk_seconds": r[9],
        "disposition": r[10],
        "summary": r[11],
        "summary_state": r[12],
    }


def _call_int(value):
    """Coerce a sender-supplied id/duration to int, or None if it isn't one."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def upsert_call_session(call_key: str, **fields) -> None:
    """Create or update the session row for a call.

    A call is reported in pieces — the transcript starts flowing before anyone
    has picked a ticket, and ended_at/talk_seconds only exist at hangup — so
    only the fields actually supplied are written. Fields that are absent, or
    explicitly None, are left alone rather than blanked: a later batch that
    omits ticket_id must not erase the ticket_id an earlier one established.
    Use set_call_summary to write the summary."""
    if not _conninfo:
        return

    cols = []
    values = []
    for col in _CALL_SESSION_FIELDS:
        value = fields.get(col)
        if value is None:
            continue
        if col in _CALL_SESSION_INT_FIELDS:
            value = _call_int(value)
            if value is None:
                continue
        elif col in _CALL_SESSION_TS_FIELDS and not isinstance(value, datetime):
            value = _parse_ts(value)
            if value is None:
                continue
        cols.append(col)
        values.append(value)

    if cols:
        placeholders = ", ".join(["%s"] * (len(cols) + 1))
        assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
        query = (
            f"INSERT INTO call_sessions (call_key, {', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT (call_key) DO UPDATE SET {assignments}"
        )
    else:
        # First segment of a call we know nothing else about yet — the row still
        # has to exist before segments can reference it.
        query = "INSERT INTO call_sessions (call_key) VALUES (%s) ON CONFLICT (call_key) DO NOTHING"

    params = (call_key, *values)

    if _use_sync:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(_sync_execute, _conninfo, query, params))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            await conn.execute(query, params)


async def save_call_segments(call_key: str, segments: list[dict]) -> int:
    """Persist a batch of transcript segments and return how many were new.

    ONE connection and one executemany for the whole batch. This is the highest
    write rate the app has, and the connection-per-query shape the rest of this
    module uses would open a Postgres connection per spoken utterance.

    Idempotent twice over: a bare ON CONFLICT DO NOTHING covers both the
    sender-supplied UUID and the (call_key, seq) uniqueness, so a retried POST
    or a segment re-sent under a fresh id is silently dropped. Segments missing
    an id, seq or text are skipped rather than failing the batch — a mangled
    utterance must not cost us the rest of the call.

    The session row must already exist (upsert_call_session first); the
    call_key foreign key is what the 60-day cascade purge hangs off."""
    if not _conninfo or not segments:
        return 0

    rows = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_id = seg.get("id")
        seq = _call_int(seg.get("seq"))
        text = seg.get("text")
        if not seg_id or seq is None or not text:
            continue
        speaker = seg.get("speaker") or "customer"
        spoken_at = _parse_ts(seg.get("spoken_at")) or datetime.now(timezone.utc)
        rows.append((seg_id, call_key, seq, speaker, text, spoken_at))

    if not rows:
        return 0

    query = (
        "INSERT INTO call_transcript_segments (id, call_key, seq, speaker, text, spoken_at) "
        "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"
    )

    if _use_sync:
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, partial(_sync_execute_many, _conninfo, query, rows))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = conn.cursor()
            await cur.executemany(query, rows)
            count = cur.rowcount

    return count if count and count > 0 else 0


async def get_call_session(call_key: str) -> dict | None:
    if not _conninfo:
        return None

    query = _CALL_SESSION_SELECT + "WHERE call_key = %s"

    if _use_sync:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, (call_key,)))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, (call_key,))
            rows = await cur.fetchall()

    if not rows:
        return None
    return _call_session_row(rows[0])


async def get_transcripts_for_calls(call_keys: list[str]) -> dict[str, list[dict]]:
    """Transcripts for several calls at once, keyed by call_key.

    One connection for the whole set. The obvious alternative — gathering
    get_call_transcript per call — opens one Postgres connection per call,
    because there is no pool in this module, and a Call tab showing twenty
    calls would then take twenty connections every time it is opened. Every
    other query in the app also takes a fresh connection, so exhausting
    max_connections here breaks chat and the ConnectWise actions too.
    """
    if not _conninfo or not call_keys:
        return {}

    query = (
        "SELECT call_key, id, seq, speaker, text, spoken_at FROM call_transcript_segments "
        "WHERE call_key = ANY(%s) ORDER BY call_key ASC, seq ASC"
    )
    params = (list(call_keys),)

    if _use_sync:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, params))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, params)
            rows = await cur.fetchall()

    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r[0], []).append(
            {
                "id": str(r[1]),
                "seq": r[2],
                "speaker": r[3],
                "text": r[4],
                "spoken_at": r[5].isoformat() if r[5] else None,
            }
        )
    return out


async def get_call_transcript(call_key: str) -> list[dict]:
    """Full transcript for one call, in spoken order (for backlog replay on the
    pod socket and for the end-of-call summary)."""
    if not _conninfo:
        return []

    query = (
        "SELECT id, seq, speaker, text, spoken_at FROM call_transcript_segments "
        "WHERE call_key = %s ORDER BY seq ASC"
    )

    if _use_sync:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, (call_key,)))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, (call_key,))
            rows = await cur.fetchall()

    return [
        {
            "id": str(r[0]),
            "seq": r[1],
            "speaker": r[2],
            "text": r[3],
            "spoken_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]


async def get_calls_for_ticket(ticket_id: int) -> list[dict]:
    """Every call attached to a ticket, oldest first. Transcripts are fetched
    per call with get_call_transcript so a ticket with a long call history
    doesn't drag every utterance back in one query."""
    if not _conninfo:
        return []

    query = _CALL_SESSION_SELECT + "WHERE ticket_id = %s ORDER BY started_at ASC, call_key ASC"

    if _use_sync:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, (ticket_id,)))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, (ticket_id,))
            rows = await cur.fetchall()

    return [_call_session_row(r) for r in rows]


async def set_call_summary(call_key: str, summary: str | None, state: str) -> None:
    """Store the generated technician notes and where they got to
    ('pending' | 'ready' | 'failed' | 'posted'). A failed generation still
    records its state, so the pod can say so instead of spinning forever."""
    if not _conninfo:
        return

    query = "UPDATE call_sessions SET summary = %s, summary_state = %s WHERE call_key = %s"
    params = (summary, state, call_key)

    if _use_sync:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(_sync_execute, _conninfo, query, params))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            await conn.execute(query, params)


async def purge_old_calls(days: int = 60) -> int:
    """Retention sweep: drop call sessions older than `days` and, by the
    segments' ON DELETE CASCADE, their transcripts with them. Returns the number
    of sessions deleted.

    Transcripts are private call audio in text form and live in this database
    only, for 60 days. The internal ConnectWise note derived from a call is
    permanent by nature; that is intended, this is not."""
    if not _conninfo:
        return 0

    query = "DELETE FROM call_sessions WHERE started_at < NOW() - (%s::int * INTERVAL '1 day')"
    params = (int(days),)

    if _use_sync:
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, partial(_sync_execute, _conninfo, query, params))
        return count
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, params)
            return cur.rowcount


# --- Admin dashboard settings (shared with the customer-facing Hercules service) ---


async def _fetch(query: str, params: tuple = ()):
    if _use_sync:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, params))
    async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchall()


async def get_admin_settings(app: str) -> list[tuple[str, str | None]]:
    """(key, value) rows for one app - overrides AND 'default.*' rows."""
    if not _conninfo:
        return []
    rows = await _fetch("SELECT key, value FROM hercules_admin_settings WHERE app = %s", (app,))
    return [(r[0], r[1]) for r in rows]


async def upsert_admin_defaults(app: str, defaults: dict[str, str]) -> None:
    """Publish this app's built-in values as 'default.<key>' rows (system-owned)."""
    if not _conninfo or not defaults:
        return
    query = (
        "INSERT INTO hercules_admin_settings (app, key, value, updated_by) "
        "VALUES (%s, %s, %s, 'system') "
        "ON CONFLICT (app, key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()"
    )
    params_seq = [(app, f"default.{k}", v) for k, v in defaults.items()]
    if _use_sync:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(_sync_execute_many, _conninfo, query, params_seq))
        return
    async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
        cur = conn.cursor()
        await cur.executemany(query, params_seq)
        await conn.commit()


# --- Hudu company mapping (see hudu_client.resolve_company) ---------------------


async def get_hudu_company(cw_company_id: int) -> dict | None:
    """The cached Hudu company for a ConnectWise company, with how old the row is."""
    if not _conninfo:
        return None
    rows = await _fetch(
        "SELECT hudu_company_id, hudu_company_name, hudu_url, "
        "EXTRACT(EPOCH FROM (NOW() - resolved_at)) FROM hudu_companies WHERE cw_company_id = %s",
        (int(cw_company_id),),
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "hudu_company_id": r[0],
        "hudu_company_name": r[1] or "",
        "hudu_url": r[2] or "",
        "age_seconds": float(r[3] or 0),
    }


async def save_hudu_company(cw_company_id: int, cw_company_name: str, mapping: dict) -> None:
    if not _conninfo:
        return
    query = (
        "INSERT INTO hudu_companies (cw_company_id, cw_company_name, hudu_company_id, "
        "hudu_company_name, hudu_url, resolved_at) VALUES (%s, %s, %s, %s, %s, NOW()) "
        "ON CONFLICT (cw_company_id) DO UPDATE SET cw_company_name = EXCLUDED.cw_company_name, "
        "hudu_company_id = EXCLUDED.hudu_company_id, hudu_company_name = EXCLUDED.hudu_company_name, "
        "hudu_url = EXCLUDED.hudu_url, resolved_at = NOW()"
    )
    params = (
        int(cw_company_id), cw_company_name or None, mapping.get("hudu_company_id"),
        mapping.get("hudu_company_name") or None, mapping.get("hudu_url") or None,
    )
    if _use_sync:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(_sync_execute, _conninfo, query, params))
        return
    async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
        await conn.execute(query, params)
