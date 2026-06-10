import os
import json
from typing import AsyncGenerator

import httpx


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    {"id": "anthropic/claude-haiku-4.5", "label": "Claude Haiku 4.5"},
    {"id": "anthropic/claude-sonnet-4", "label": "Claude Sonnet 4"},
    {"id": "openai/gpt-4o", "label": "GPT-4o"},
    {"id": "openai/gpt-4o-mini", "label": "GPT-4o Mini"},
    {"id": "google/gemini-2.5-flash-preview", "label": "Gemini 2.5 Flash"},
]


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
        "X-Title": "NTM AI Assistant",
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
                    await response.aread()
                    yield json.dumps({"error": f"AI service error ({response.status_code})"})
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


async def generate_resolution_note(source_text: str, model: str) -> str:
    system_prompt = (
        "You are a technical note writer for an MSP ticketing system. "
        "Write a resolution note for this support ticket based on the source material below. "
        "The source may be a technician chat transcript OR raw internal ticket notes — "
        "treat internal commentary as reference only; do not copy it verbatim.\n\n"
        "RULES (the resolution note may be visible to the customer — treat it as customer-facing):\n"
        "- Write a concise paragraph summary of what the issue was, what was done, and the outcome\n"
        "- NEVER mention pricing, costs, billing, or fees\n"
        "- NEVER admit fault, wrongdoing, or blame anyone (customer, technician, vendor, or third party)\n"
        "- NEVER mention vendor names, internal tools, internal ticket IDs, or internal team discussions\n"
        "- NEVER include diagnostic asides, speculation, frustration, or raw log/error dumps\n"
        "- Omit anything inappropriate for the customer to read\n"
        "- Keep it professional and solution-focused\n"
        "- Write in past tense\n"
        "- End the note with a new line: STATUS: Resolved\n\n"
        "Format: Start with '[Resolution - CW Chat Pod]' on the first line, "
        "then the paragraph summary, then 'STATUS: Resolved' on the last line."
    )

    return await _call_openrouter(
        system_prompt,
        f"Write a resolution note based on this support context:\n\n{source_text}",
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
