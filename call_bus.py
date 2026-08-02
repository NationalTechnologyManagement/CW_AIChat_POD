r"""Call-transcript hub: a Redis pub/sub bus + per-process WebSocket fan-out.

Modelled on live.py, but deliberately a SEPARATE bus. Transcript segments are
published to `call:ticket:<id>`; a background subscriber fans them out to the
WebSocket clients connected to THIS process. With multiple replicas, each one
runs its own subscriber and serves its own clients — Redis handles the spread.

If REDIS_URL is unset (local single-instance dev), publish_segment() degrades to
a direct local fan-out, so the pod still works without a bus.

Unlike live.py this module does NOT persist: /call/ingest writes the segments to
Postgres (idempotently, by UUID) BEFORE publishing, and only publishes the rows
it actually won the insert on. So the bus carries exactly the new segments, and
the Redis and no-Redis paths deliver the same thing.

    ############################################################################
    # DO NOT MERGE THIS BUS INTO live.py, AND NEVER PUBLISH A TRANSCRIPT ONTO
    # "live:ticket:*".
    #
    # Hercules-Client (C:\Github\Hercules-Client) is the CUSTOMER-FACING widget
    # service. Its server/lib/live-bridge.js psubscribes "live:ticket:*" and its
    # fanout() relays EVERY envelope it receives, verbatim, to the customer's
    # browser — there is no kind/type allowlist and no filtering of any sort.
    #
    # One transcript envelope on the wrong channel therefore streams a
    # technician's private call audio, live, to the person on the other end of
    # the phone. That is why this file has its own channel prefix, its own Redis
    # client, its own subscriber loop and its own client registry, duplicated
    # rather than shared. The duplication IS the safety property. Keep it.
    ############################################################################
"""
import asyncio
import json
import os

import redis.asyncio as redis_async

# The transcript channel. NEVER "live:ticket:" — see the warning above.
CHANNEL_PREFIX = "call:ticket:"

_redis: "redis_async.Redis | None" = None
_sub_task: "asyncio.Task | None" = None

# ticket_id (int) -> set of connected WebSockets on THIS process.
# Its OWN registry: sharing live.py's would fan transcripts out to /live/ws.
_clients: dict[int, set] = {}


def has_clients(ticket_id: int) -> bool:
    """Whether any WebSocket (i.e. a technician tab watching the transcript) is
    connected to this ticket on THIS process."""
    return bool(_clients.get(int(ticket_id)))


def _channel(ticket_id: int) -> str:
    return f"{CHANNEL_PREFIX}{ticket_id}"


async def start() -> None:
    global _redis, _sub_task
    url = os.getenv("REDIS_URL")
    if not url:
        print("[call] REDIS_URL not set — running single-instance (no cross-replica bus)")
        return
    _redis = redis_async.from_url(url, decode_responses=True)
    _sub_task = asyncio.create_task(_subscriber_loop())
    print("[call] Redis bus connected")


async def stop() -> None:
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


async def publish_segment(ticket_id: "int | None", envelope: dict) -> None:
    """Put a transcript segment on the bus. With Redis, the subscriber fans it
    out (so it reaches every replica). Without Redis, do it inline.

    A call can start before its ticket exists, so ticket_id may be None — there
    is no channel to route to and nobody can be watching, so the segment is
    dropped from the bus. It is already in Postgres and /call/ws will serve it
    in the backlog once the call is attached to a ticket.

    Never raises: a bus outage must not fail an ingest POST. The voice agent is
    on the phone with a customer; a 500 there is worse than a missing bubble."""
    if ticket_id is None:
        return
    if _redis is not None:
        try:
            await _redis.publish(_channel(ticket_id), json.dumps(envelope))
        except Exception as e:
            print(f"[call] publish failed for ticket {ticket_id}: {e}")
    else:
        try:
            await _handle_envelope(envelope)
        except Exception as e:
            print(f"[call] inline handle failed for ticket {ticket_id}: {e}")


async def _handle_envelope(env: dict) -> None:
    """Deliver to local WebSocket clients. Persistence already happened in
    /call/ingest, so there is nothing to write here."""
    try:
        ticket_id = int(env.get("ticketId"))
    except (TypeError, ValueError):
        return
    await _fanout(ticket_id, env)


async def _fanout(ticket_id: int, env: dict) -> None:
    conns = list(_clients.get(int(ticket_id), ()))
    for ws in conns:
        try:
            await ws.send_json(env)
        except Exception:
            unregister(ticket_id, ws)


async def _subscriber_loop() -> None:
    pattern = f"{CHANNEL_PREFIX}*"
    while True:
        try:
            pubsub = _redis.pubsub()
            await pubsub.psubscribe(pattern)
            print(f"[call] subscribed to {pattern}")
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
            print(f"[call] subscriber error, retrying in 2s: {e}")
            await asyncio.sleep(2)
