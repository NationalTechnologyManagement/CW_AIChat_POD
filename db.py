import asyncio
import json
import os
import sys
from functools import partial

import psycopg

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
