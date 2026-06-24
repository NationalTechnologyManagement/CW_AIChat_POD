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
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_live_messages_ticket ON live_messages (ticket_id, created_at, id);

CREATE TABLE IF NOT EXISTS live_sessions (
    ticket_id   INTEGER PRIMARY KEY,
    status      VARCHAR(16) NOT NULL DEFAULT 'active',  -- 'active' | 'ended'
    started_by  TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at    TIMESTAMPTZ
);
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
) -> bool:
    """Persist one live message. Idempotent: a duplicate id (same message seen by
    another subscriber/replica) is silently ignored. Returns True if a row was
    actually inserted (i.e. this was the first time we saw this id).

    created_at is stored from the envelope's send-time `ts` so the transcript
    orders by when messages were sent, not when this replica happened to persist
    them (which can differ under network jitter / multiple subscribers)."""
    if not _conninfo:
        return False

    created_at = _parse_ts(ts) or datetime.now(timezone.utc)
    query = (
        "INSERT INTO live_messages (id, ticket_id, sender, member_identifier, author_name, body, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING"
    )
    params = (msg_id, ticket_id, sender, member_identifier, author_name, body, created_at)

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
        "SELECT id, sender, member_identifier, author_name, body, created_at "
        "FROM live_messages WHERE ticket_id = %s ORDER BY created_at ASC, id ASC"
    )

    if _use_sync:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, partial(_sync_query, _conninfo, query, (ticket_id,)))
    else:
        async with await psycopg.AsyncConnection.connect(_conninfo) as conn:
            cur = await conn.execute(query, (ticket_id,))
            rows = await cur.fetchall()

    return [
        {
            "id": str(r[0]),
            "sender": r[1],
            "memberIdentifier": r[2],
            "authorName": r[3],
            "body": r[4],
            "ts": r[5].isoformat() if r[5] else None,
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
