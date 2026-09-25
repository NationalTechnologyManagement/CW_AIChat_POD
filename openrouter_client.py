import os
import json
import re
import time
from typing import AsyncGenerator

import httpx


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# The model every request falls back to when none is given, and the one the
# pod's dropdown pre-selects (the dashboard's default_model can override that).
# GPT-6 Luna is OpenAI's fast/cheap tier — the Haiku-class slot in their lineup.
DEFAULT_MODEL = "openai/gpt-6-luna"

# Used when the OpenRouter catalog can't be fetched; also always allowed so
# saved defaults keep working even if a model drops out of a refreshed list.
# Keep this in the same order as MODEL_SLOTS — the first entry is the default.
FALLBACK_MODELS = [
    {"id": DEFAULT_MODEL, "label": "GPT-6 Luna"},
    {"id": "anthropic/claude-haiku-4.5", "label": "Claude Haiku 4.5"},
    {"id": "anthropic/claude-sonnet-5", "label": "Claude Sonnet 5"},
    {"id": "anthropic/claude-opus-5.5", "label": "Claude Opus 5.5"},
    {"id": "openai/gpt-6-sol", "label": "GPT-6 Sol"},
    {"id": "openai/gpt-5.4-mini", "label": "GPT-5.4 Mini"},
    {"id": "google/gemini-3.8-flash", "label": "Gemini 3.8 Flash"},
]

# One dropdown entry per slot, filled with the newest vision-capable model
# whose id matches. Patterns deliberately exclude -fast/-pro/-chat/preview/:free
# variants so the list stays curated while versions update themselves.
# The first slot is the dropdown default. GPT-6 ships as named tiers (Luna =
# fast/cheap, Sol = mid, Astra = top) rather than a plain "gpt-6", so the two
# OpenAI flagship slots track the Luna and Sol tiers by name.
MODEL_SLOTS = [
    re.compile(r"^openai/gpt-[\d.]+-luna$"),
    re.compile(r"^anthropic/claude-haiku-[\d.]+$"),
    re.compile(r"^anthropic/claude-sonnet-[\d.]+$"),
    re.compile(r"^anthropic/claude-opus-[\d.]+$"),
    re.compile(r"^openai/gpt-[\d.]+-sol$"),
    re.compile(r"^openai/gpt-[\d.]+-mini$"),
    re.compile(r"^google/gemini-[\d.]+-flash$"),
]

MODELS_TTL_SECONDS = 6 * 3600

_models_cache: dict = {"models": [], "ids": set(), "no_tool_ids": set(), "fetched_at": 0.0}


async def refresh_models() -> None:
    """Refresh the model list from OpenRouter's catalog. Never raises."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(OPENROUTER_MODELS_URL)
            response.raise_for_status()
            catalog = response.json().get("data", [])
    except Exception as e:
        print(f"[models] Refresh failed, using previous/fallback list: {e}")
        return

    vision_models = [
        m for m in catalog
        if "image" in (m.get("architecture") or {}).get("input_modalities", [])
    ]

    models = []
    no_tools = set()
    for pattern in MODEL_SLOTS:
        candidates = [m for m in vision_models if pattern.match(m["id"])]
        if not candidates:
            continue
        newest = max(candidates, key=lambda m: m.get("created", 0))
        # OpenRouter names look like "Anthropic: Claude Opus 4.8" — drop the vendor prefix
        label = (newest.get("name") or newest["id"]).split(": ", 1)[-1]
        models.append({"id": newest["id"], "label": label})
        # Only a catalog row that lists its parameters AND omits "tools" counts
        # as a refusal. A row with no parameter list at all is unknown, not no.
        supported = newest.get("supported_parameters")
        if supported and "tools" not in supported:
            no_tools.add(newest["id"])

    if models:
        _models_cache["models"] = models
        _models_cache["ids"] = {m["id"] for m in models}
        _models_cache["no_tool_ids"] = no_tools
        _models_cache["fetched_at"] = time.time()
        print(f"[models] Refreshed: {[m['id'] for m in models]}"
              + (f" (no tool support: {sorted(no_tools)})" if no_tools else ""))


async def get_models() -> list[dict]:
    """Current model list for the UI; refreshes when stale, falls back if empty."""
    if time.time() - _models_cache["fetched_at"] > MODELS_TTL_SECONDS:
        await refresh_models()
    return _models_cache["models"] or FALLBACK_MODELS


def is_model_allowed(model_id: str) -> bool:
    return model_id in _models_cache["ids"] or any(m["id"] == model_id for m in FALLBACK_MODELS)


def model_supports_tools(model_id: str) -> bool:
    """Whether this model can use ConnectWise tools.

    Fail open: only a catalog entry that explicitly lists its parameters without
    "tools" is treated as a no. Silently dropping the tools is worse than trying
    them — it puts Hercules back to insisting it can't see other tickets.
    """
    return model_id not in _models_cache["no_tool_ids"]


def content_to_text(content) -> str:
    """Flatten structured (multimodal) message content to plain text.

    Image parts become an "[N image(s) attached]" marker so text-only
    pipelines (summaries, resolution notes, emails) stay coherent.
    """
    if isinstance(content, str):
        return content
    texts = []
    image_count = 0
    for part in content:
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            image_count += 1
    if image_count:
        texts.append(f"[{image_count} image(s) attached]")
    return "\n".join(t for t in texts if t)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://trustntm.com",
        "X-Title": "Hercules (NTM AI Assistant)",
    }


def _accumulate_tool_call(calls: dict, tc: dict) -> None:
    """Fold one streamed tool-call delta into the accumulator, keyed by index.

    The wire protocol sends `id` and `function.name` once, on the first delta
    for an index, then `function.arguments` in fragments. So the name is
    ASSIGNED and the arguments are strictly CONCATENATED — anything cleverer
    (skipping a fragment that repeats the accumulated tail, say) silently
    corrupts JSON like {"text":"Hello""} where a lone quote is a real fragment.
    """
    index = tc.get("index")
    if not isinstance(index, int):
        call_id = tc.get("id")
        if call_id:
            index = next((i for i, c in calls.items() if c["id"] == call_id), len(calls))
        else:
            # No index and no id: this is a continuation of the call in flight,
            # not a new one — starting a fresh slot would strand the fragment.
            index = max(calls) if calls else 0
    slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
    if tc.get("id"):
        slot["id"] = tc["id"]
    function = tc.get("function") or {}
    if function.get("name"):
        slot["name"] = function["name"]
    if isinstance(function.get("arguments"), str):
        slot["arguments"] += function["arguments"]


async def stream_chat(
    messages: list[dict],
    model: str,
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    max_tokens: int = 2048,
) -> AsyncGenerator[dict, None]:
    """Stream one model turn, yielding events:

        {"content": "..."}      a text delta, as it arrives
        {"tool_calls": [...]}   the turn ended asking for tools (accumulated,
                                each {id, name, arguments}) — emitted once
        {"error": "..."}        terminal; nothing else follows
        {"done": True}          the turn finished cleanly

    The caller decides what to do with tool calls; this function never executes
    anything. `tools` must be resent on every round of a tool conversation —
    pass tool_choice="none" to hand control back rather than dropping them.
    """
    full_messages = list(messages)
    if system_prompt:
        full_messages = [{"role": "system", "content": system_prompt}] + full_messages

    payload = {
        "model": model,
        "messages": full_messages,
        "stream": True,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    calls: dict[int, dict] = {}
    finish_reason = None

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with client.stream(
                "POST", OPENROUTER_URL, headers=_headers(), json=payload,
            ) as response:
                if response.status_code == 402:
                    yield {"error": "OpenRouter credits exhausted — check your account balance"}
                    return
                if response.status_code == 429:
                    yield {"error": "Rate limited — try again in a moment"}
                    return
                if response.status_code >= 400:
                    body = await response.aread()
                    print(f"[chat] OpenRouter error {response.status_code}: {body[:500]!r}")
                    detail = ""
                    try:
                        detail = json.loads(body)["error"]["message"]
                    except Exception:
                        pass
                    message = f"AI service error ({response.status_code})"
                    if detail:
                        message += f": {detail}"
                    yield {"error": message}
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # A provider failing mid-stream still returns HTTP 200 — the
                    # error rides in the chunk, so it has to be checked here or
                    # the tech just sees the answer stop.
                    if chunk.get("error"):
                        detail = chunk["error"].get("message") if isinstance(chunk["error"], dict) else str(chunk["error"])
                        print(f"[chat] mid-stream error: {detail!r}")
                        yield {"error": f"AI service error: {str(detail)[:180]}"}
                        return
                    # The final usage chunk carries an empty choices list.
                    choice = (chunk.get("choices") or [{}])[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    # Streaming sends "delta"; a provider that answers a stream
                    # request in one shot sends "message" instead.
                    delta = choice.get("delta") or choice.get("message") or {}
                    content = delta.get("content")
                    if content:
                        yield {"content": content}
                    for tc in (delta.get("tool_calls") or []):
                        _accumulate_tool_call(calls, tc)

        except httpx.ReadTimeout:
            yield {"error": "Response timed out — try again"}
            return
        except httpx.ConnectError:
            yield {"error": "Could not connect to AI service"}
            return
        except httpx.HTTPError as e:
            print(f"[chat] stream failed: {e!r}")
            yield {"error": "Connection to the AI service dropped — try again"}
            return

    if finish_reason == "length" and calls:
        # Truncated mid-arguments: the JSON is unparseable and "acting" on it
        # would mean acting on a half-written note. Fail loudly instead.
        yield {"error": "The response was cut off before the action was fully written — try again."}
        return
    if calls:
        finished, seen_ids = [], set()
        for i in sorted(calls):
            call = calls[i]
            if not call["name"]:
                continue
            # Each call is answered by id, so they have to be present and
            # distinct — synthesize one when a provider omits or repeats it.
            if not call["id"] or call["id"] in seen_ids:
                call["id"] = f"tc_{i}"
            seen_ids.add(call["id"])
            call["arguments"] = call["arguments"].strip() or "{}"
            finished.append(call)
        if finished:
            yield {"tool_calls": finished}
    yield {"done": True, "finish_reason": finish_reason}


async def summarize_chat(messages: list[dict], model: str) -> str:
    chat_text = "\n".join(
        f"{m['role'].upper()}: {content_to_text(m['content'])}" for m in messages
    )

    system_prompt = (
        "You are a technical note writer for an MSP ticketing system. "
        "Summarize this support chat into a structured internal ticket note. "
        "Use EXACTLY this format:\n\n"
        "[AI Analysis - CW Chat Pod]\n\n"
        "INVESTIGATION:\n"
        "- What was looked into or asked about\n\n"
        "FINDINGS:\n"
        "- What was discovered or determined\n\n"
        "ACTIONS RECOMMENDED:\n"
        "- Specific next steps or commands to run\n\n"
        "STATUS: [In Progress / Waiting on Client / Escalation Needed / Resolved]\n\n"
        "RESOLUTION: [Only include if a resolution was reached, otherwise omit this section]\n\n"
        "Rules: Write in past tense. Be concise — bullet points, not paragraphs. "
        "Do not include conversational filler. Only include sections that have content."
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers=_headers(),
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Summarize this support chat into internal ticket notes:\n\n{chat_text}",
                    },
                ],
                "max_tokens": 1024,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def summarize_live_chat(messages: list[dict], model: str) -> str:
    """Summarize a live technician<->customer chat into one internal ticket note.

    `messages` are live_messages rows ({sender, authorName, body, ...}); unlike
    summarize_chat this is a HUMAN conversation, so the note is labeled as a live
    chat transcript, not AI analysis.
    """
    transcript = "\n".join(
        f"{(m.get('authorName') or m.get('sender') or 'Unknown')} "
        f"({'Technician' if m.get('sender') == 'technician' else 'Customer'}): "
        f"{m.get('body', '')}"
        for m in messages
        if m.get("sender") in ("technician", "customer")
    )

    system_prompt = (
        "You are a technical note writer for an MSP ticketing system. "
        "Summarize this LIVE chat between a technician and a customer into a "
        "structured internal ticket note. Use EXACTLY this format:\n\n"
        "[Live Chat Summary - CW Chat Pod]\n\n"
        "ISSUE:\n- What the customer reported\n\n"
        "DISCUSSION:\n- Key points covered during the live chat\n\n"
        "OUTCOME / NEXT STEPS:\n- What was resolved or what happens next\n\n"
        "STATUS: [In Progress / Waiting on Client / Escalation Needed / Resolved]\n\n"
        "Rules: Write in past tense. Be concise — bullet points, not paragraphs. "
        "No conversational filler. Only include sections that have content."
    )

    return await _call_openrouter(
        system_prompt,
        f"Summarize this live support chat into an internal ticket note:\n\n{transcript}",
        model,
    )


async def _call_openrouter(system_prompt: str, user_content: str, model: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers=_headers(),
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def extract_search_keywords(ticket_summary: str) -> list[str]:
    """Extract 2-4 technical search keywords from a ticket summary using the cheap default model."""
    result = await _call_openrouter(
        system_prompt=(
            "Extract the core technical keywords from this IT support ticket summary. "
            "Return ONLY a comma-separated list of 2-4 specific technical terms that describe the issue. "
            "Focus on: device types, software names, error types, specific symptoms. "
            "Exclude: company names, locations, people names, generic words like 'issue' or 'problem'.\n"
            "Examples:\n"
            "- 'BLEZ | Intermittent Scanner Issues at The Crossing' → scanner, intermittent, scanning\n"
            "- 'Need to quote updated firewall for Grace' → firewall\n"
            "- 'Outlook keeps crashing when opening attachments' → outlook, crashing, attachments\n"
            "- 'VPN disconnects randomly throughout the day' → vpn, disconnects\n"
            "Reply with ONLY the comma-separated keywords, nothing else."
        ),
        user_content=ticket_summary,
        model=DEFAULT_MODEL,
        max_tokens=50,
        temperature=0,
    )
    # Parse comma-separated response into clean keyword list
    keywords = [k.strip().lower() for k in result.split(",") if k.strip()]
    # Filter out anything too short or suspiciously long (not a keyword)
    return [k for k in keywords if 2 < len(k) < 30]


async def generate_internal_resolution_note(source_text: str, model: str) -> str:
    """Internal technician resolution note — NOT shown to the customer.

    Goes into the ticket's Internal Analysis tab and the resolving time entry's
    notes, so it can include the technical specifics a tech needs on the record.
    """
    system_prompt = (
        "You are a technical note writer for an MSP ticketing system. "
        "Write an INTERNAL technician resolution note from the source material "
        "(a technician chat transcript OR raw internal ticket notes). "
        "This note is INTERNAL ONLY — it is never shown to the customer, so include "
        "the technical specifics another tech would need.\n\n"
        "Use EXACTLY this format:\n\n"
        "[Resolution - CW Chat Pod]\n\n"
        "ISSUE:\n- What the problem was\n\n"
        "ROOT CAUSE:\n- Why it happened (omit this section if not determined)\n\n"
        "RESOLUTION:\n- What was done to fix it — specific steps, commands, console paths, settings\n\n"
        "STATUS: Resolved\n\n"
        "Rules: Write in past tense. Be concise — bullet points, not paragraphs. "
        "No conversational filler. Only include sections that have content."
    )

    return await _call_openrouter(
        system_prompt,
        f"Write an internal resolution note based on this support context:\n\n{source_text}",
        model,
    )


async def generate_time_entry_note(source_text: str, model: str, ticket_summary: str) -> str:
    """The technician's work note for a time entry — what they actually did.

    Shorter and plainer than a resolution note: it lands in the time entry (and
    the ticket's Internal Analysis), where the next tech reads it as the work
    log. No headers, no template — just the work.
    """
    system_prompt = (
        "You write the work note a technician puts on a ConnectWise time entry at an MSP. "
        "From the source material (a technician's chat with an AI assistant, or the ticket's "
        "own notes), write the note describing the work THIS technician just did.\n\n"
        "Rules:\n"
        "- 1-4 short bullet points, or two plain sentences — nothing longer\n"
        "- Past tense, factual, technician voice ('Rebuilt the Outlook profile...')\n"
        "- Include the specifics another tech would need: what was checked, what was changed, "
        "commands run, error messages, the outcome\n"
        "- If the issue is not finished, end with the current state / what is next\n"
        "- No headers, no labels, no '[AI Analysis]' banner, no conversational filler\n"
        "- Do not invent work that is not in the source material. If the source shows only "
        "discussion and no action taken, describe the investigation instead\n"
        "- Output ONLY the note text"
    )

    return await _call_openrouter(
        system_prompt,
        f"Ticket: {ticket_summary}\n\nSource material:\n\n{source_text}",
        model,
        max_tokens=500,
    )


async def generate_customer_email(
    source_text: str, model: str, ticket_summary: str, contact_name: str,
    purpose: str = "resolution",
) -> str:
    """A customer-facing email body.

    purpose="resolution" summarizes a completed fix (used by Resolve);
    purpose="update" is a mid-ticket progress note that must NOT claim the issue
    is finished (used by Add Time).
    """
    if purpose == "update":
        intent = (
            "write a short progress update to the customer on work that is STILL IN PROGRESS.\n\n"
            "- Never say the issue is resolved, fixed, or closed — this is an update, not a resolution\n"
            "- Say what has been done so far and what happens next\n"
            "- If something is needed from the customer, ask for it plainly in one line\n\n"
        )
    else:
        intent = "write a clean email to the customer summarizing what was done.\n\n"

    system_prompt = (
        "You are writing a professional customer-facing email for an MSP (managed IT services provider). "
        "Based on the source material below (a technician chat OR raw internal ticket notes), "
        + intent +
        "STRICT RULES — VIOLATING THESE IS UNACCEPTABLE:\n"
        "- NEVER mention pricing, costs, billing, fees, or charges\n"
        "- NEVER admit fault, wrongdoing, or blame anyone\n"
        "- NEVER mention internal team discussions, vendor names, or vendor-specific issues\n"
        "- NEVER copy raw internal notes, diagnostic asides, speculation, or log/error dumps\n"
        "- NEVER use overly technical jargon the customer wouldn't understand\n"
        "- NEVER mention ticket numbers, internal systems, or tools used\n"
        "- Treat all input as internal source material — extract only what is safe and appropriate for the customer to read\n\n"
        "FORMAT:\n"
        "- Start with a greeting using the contact's first name\n"
        "- Write a short professional paragraph or bulleted summary of what was done and the outcome\n"
        "- Keep it to 1-2 short paragraphs or a brief bullet list — no long emails\n"
        "- End with a brief offer to help if they need anything else\n"
        "- Do NOT include a subject line, sign-off name, or email headers — just the body text\n"
        "- Tone: professional, friendly, solution-focused"
    )

    return await _call_openrouter(
        system_prompt,
        f"Contact name: {contact_name}\nTicket summary: {ticket_summary}\n\nInternal source material to summarize for the customer:\n\n{source_text}",
        model,
        temperature=0.4,
    )
