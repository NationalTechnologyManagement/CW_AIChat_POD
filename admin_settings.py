"""Runtime settings the Hercules admin dashboard can change for THIS pod.

The dashboard is served by the customer-facing Hercules service (/admin); both
services share one Postgres, so it writes rows into `hercules_admin_settings`
(app = 'internal') and this module reads them back:

    default_model      - model the pod's dropdown pre-selects (and is always allowed)
    prompt_role        - the role/audience paragraph at the top of the system prompt
    prompt_guidelines  - the static GUIDELINES bullets

Reads are served from a small in-process cache so build_system_prompt stays
synchronous; `ensure_fresh()` is awaited once per request before the prompt is
built. On boot the pod publishes its built-in values under `default.<key>` so
the dashboard can show "customized vs default" and offer a reset.

Everything here degrades to the built-in defaults: no DATABASE_URL, a missing
table, or a DB blip never breaks a chat.
"""

import time

import db

APP = "internal"
TTL_SECONDS = 30.0

_cache: dict[str, str] = {}
_loaded_at: float = 0.0


def get(key: str, default: str | None = None) -> str | None:
    """Sync read of an admin override; `default` when none is set."""
    value = _cache.get(key)
    return value if value else default


def is_default_model(model_id: str) -> bool:
    return bool(model_id) and model_id == _cache.get("default_model")


def with_default_model(models: list[dict]) -> list[dict]:
    """Reorder the dropdown so the dashboard-chosen default comes first (added
    if the curated catalog doesn't list it)."""
    chosen = _cache.get("default_model")
    if not chosen:
        return models
    rest = [m for m in models if m["id"] != chosen]
    match = next((m for m in models if m["id"] == chosen), None)
    return [match or {"id": chosen, "label": chosen}] + rest


async def refresh() -> None:
    global _cache, _loaded_at
    try:
        rows = await db.get_admin_settings(APP)
    except Exception as e:  # noqa: BLE001 - never let settings break a request
        print(f"[admin-settings] refresh failed, keeping previous values: {e}")
        _loaded_at = time.time()
        return
    _cache = {k: v for k, v in rows if v and not k.startswith("default.")}
    _loaded_at = time.time()


async def ensure_fresh() -> None:
    if time.time() - _loaded_at > TTL_SECONDS:
        await refresh()


async def init(defaults: dict[str, str]) -> None:
    """Boot: load overrides and publish this pod's built-in defaults."""
    await refresh()
    try:
        await db.upsert_admin_defaults(APP, {k: v for k, v in defaults.items() if v})
    except Exception as e:  # noqa: BLE001
        print(f"[admin-settings] could not publish defaults: {e}")
