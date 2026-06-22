import os
import json
import re
import time
from typing import AsyncGenerator

import httpx


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Used when the OpenRouter catalog can't be fetched; also always allowed so
# saved defaults keep working even if a model drops out of a refreshed list.
FALLBACK_MODELS = [
    {"id": "anthropic/claude-haiku-4.5", "label": "Claude Haiku 4.5"},
    {"id": "anthropic/claude-sonnet-4.6", "label": "Claude Sonnet 4.6"},
    {"id": "anthropic/claude-opus-4.8", "label": "Claude Opus 4.8"},
    {"id": "openai/gpt-5.5", "label": "GPT-5.5"},
    {"id": "openai/gpt-5.4-mini", "label": "GPT-5.4 Mini"},
    {"id": "google/gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
]

# One dropdown entry per slot, filled with the newest vision-capable model
# whose id matches. Patterns deliberately exclude -fast/-pro/-chat/preview/:free
# variants so the list stays curated while versions update themselves.
MODEL_SLOTS = [
    re.compile(r"^anthropic/claude-haiku-[\d.]+$"),
    re.compile(r"^anthropic/claude-sonnet-[\d.]+$"),
    re.compile(r"^anthropic/claude-opus-[\d.]+$"),
    re.compile(r"^openai/gpt-[\d.]+$"),
    re.compile(r"^openai/gpt-[\d.]+-mini$"),
    re.compile(r"^google/gemini-[\d.]+-flash$"),
]

MODELS_TTL_SECONDS = 6 * 3600

_models_cache: dict = {"models": [], "ids": set(), "fetched_at": 0.0}


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
    for pattern in MODEL_SLOTS:
        candidates = [m for m in vision_models if pattern.match(m["id"])]
        if not candidates:
            continue
        newest = max(candidates, key=lambda m: m.get("created", 0))
        # OpenRouter names look like "Anthropic: Claude Opus 4.8" — drop the vendor prefix
        label = (newest.get("name") or newest["id"]).split(": ", 1)[-1]
        models.append({"id": newest["id"], "label": label})

    if models:
        _models_cache["models"] = models
        _models_cache["ids"] = {m["id"] for m in models}
        _models_cache["fetched_at"] = time.time()
        print(f"[models] Refreshed: {[m['id'] for m in models]}")


async def get_models() -> list[dict]:
    """Current model list for the UI; refreshes when stale, falls back if empty."""
    if time.time() - _models_cache["fetched_at"] > MODELS_TTL_SECONDS:
        await refresh_models()
    return _models_cache["models"] or FALLBACK_MODELS


def is_model_allowed(model_id: str) -> bool:
    return model_id in _models_cache["ids"] or any(m["id"] == model_id for m in FALLBACK_MODELS)


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


async def stream_chat(
    messages: list[dict],
    model: str,
    system_prompt: str,
) -> AsyncGenerator[str, None]:
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with client.stream(
                "POST",
                OPENROUTER_URL,
                headers=_headers(),
                json={
                    "model": model,
                    "messages": full_messages,
                    "stream": True,
                    "max_tokens": 2048,
                },
            ) as response:
                if response.status_code == 402:
                    yield json.dumps({"error": "OpenRouter credits exhausted — check your account balance"})
                    return
                if response.status_code == 429:
                    yield json.dumps({"error": "Rate limited — try again in a moment"})
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
                    yield json.dumps({"error": message})
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        yield json.dumps({"done": True})
                        return
                    try:
                        chunk = json.loads(data)
                        content = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if content:
                            yield json.dumps({"content": content})
                    except json.JSONDecodeError:
                        continue

        except httpx.ReadTimeout:
            yield json.dumps({"error": "Response timed out — try again"})
        except httpx.ConnectError:
            yield json.dumps({"error": "Could not connect to AI service"})


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
    """Extract 2-4 technical search keywords from a ticket summary using Haiku (cheap + fast)."""
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
        model="anthropic/claude-haiku-4.5",
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


async def generate_customer_email(
    source_text: str, model: str, ticket_summary: str, contact_name: str
) -> str:
    system_prompt = (
        "You are writing a professional customer-facing email for an MSP (managed IT services provider). "
        "Based on the source material below (a technician chat OR raw internal ticket notes), "
        "write a clean email to the customer summarizing what was done.\n\n"
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
