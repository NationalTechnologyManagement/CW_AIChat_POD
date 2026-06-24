"""Live messaging hub: a Redis pub/sub bus + per-process WebSocket fan-out.

The pod is the single writer to Postgres. Every live message (from this pod's
technicians OR from Hercules customers) is published to `live:ticket:<id>`; a
background subscriber persists it (idempotently, by UUID) and fans it out to the
WebSocket clients connected to THIS process. With multiple replicas, each one
runs its own subscriber and serves its own clients — Redis handles the spread.

If REDIS_URL is unset (local single-instance dev), publish() degrades to a
direct persist + local fan-out, so the pod still works without a bus.
"""
import asyncio
import json
import os

import redis.asyncio as redis_async

import db

_redis: "redis_async.Redis | None" = None
_sub_task: "asyncio.Task | None" = None

# ticket_id (int) -> set of connected WebSockets on THIS process
_clients: dict[int, set] = {}


def _channel(ticket_id: int) -> str:
    return f"live:ticket:{ticket_id}"


async def init_live() -> None:
    global _redis, _sub_task
    url = os.getenv("REDIS_URL")
    if not url:
        print("[live] REDIS_URL not set — running single-instance (no cross-service bus)")
        return
    _redis = redis_async.from_url(url, decode_responses=True)
    _sub_task = asyncio.create_task(_subscriber_loop())
    print("[live] Redis bus connected")


async def close_live() -> None:
    global _redis, _sub_task
    if _sub_task:
        _sub_task.cancel()
        try:
            await _sub_task
        except asyncio.CancelledError:
            pass
        _sub_task = None
    if _redis:
        await _redis.aclose()
        _redis = None


def register(ticket_id: int, ws) -> None:
    _clients.setdefault(int(ticket_id), set()).add(ws)


def unregister(ticket_id: int, ws) -> None:
    conns = _clients.get(int(ticket_id))
    if conns:
        conns.discard(ws)
        if not conns:
            _clients.pop(int(ticket_id), None)


async def publish(ticket_id: int, envelope: dict) -> None:
    """Put a message on the bus. With Redis, the subscriber persists + fans out
    (so it reaches every replica). Without Redis, do it inline.

    Never raises: a bus outage must not abort callers (e.g. /live/end still needs
    to write its summary note even if the live_end broadcast fails)."""
    if _redis is not None:
        try:
            await _redis.publish(_channel(ticket_id), json.dumps(envelope))
        except Exception as e:
            print(f"[live] publish failed for ticket {ticket_id}: {e}")
    else:
        try:
            await _handle_envelope(envelope)
        except Exception as e:
            print(f"[live] inline handle failed for ticket {ticket_id}: {e}")


async def _handle_envelope(env: dict) -> None:
    """Persist (real messages only) then deliver to local WebSocket clients."""
    ticket_id = int(env.get("ticketId"))
    if env.get("kind") == "message":
        try:
            await db.save_live_message(
                msg_id=env["id"],
                ticket_id=ticket_id,
                sender=env.get("sender", "system"),
                body=env.get("body", ""),
                member_identifier=env.get("memberIdentifier"),
                author_name=env.get("authorName"),
                ts=env.get("ts"),
            )
        except Exception as e:
            print(f"[live] persist failed for ticket {ticket_id}: {e}")
    await _fanout(ticket_id, env)


async def _fanout(ticket_id: int, env: dict) -> None:
    conns = list(_clients.get(int(ticket_id), ()))
    for ws in conns:
        try:
            await ws.send_json(env)
        except Exception:
            unregister(ticket_id, ws)


async def _subscriber_loop() -> None:
    while True:
        try:
            pubsub = _redis.pubsub()
            await pubsub.psubscribe("live:ticket:*")
            print("[live] subscribed to live:ticket:*")
            async for message in pubsub.listen():
                if message.get("type") != "pmessage":
                    continue
                try:
                    env = json.loads(message["data"])
                except (ValueError, TypeError):
                    continue
                await _handle_envelope(env)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[live] subscriber error, retrying in 2s: {e}")
            await asyncio.sleep(2)
