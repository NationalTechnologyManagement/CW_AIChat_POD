# Hercules Internal

Technician-facing Hercules pod for ConnectWise Manage.

## What Hercules can do in ConnectWise

The chat assistant is wired into the ConnectWise API, not just reading the
current ticket.

**Lookups it performs on its own** (read-only, no confirmation):

- `search_tickets` — searches *every* service ticket in the instance, scoped to
  the contact, the company, or everything. This is what answers "have we seen
  this before".
- `get_ticket_details` — reads another ticket's notes **and its time entries**.
  The work log matters: ConnectWise stores time-entry notes on the entry, not as
  ticket notes, so most of "what did we actually do" lives there.
- `list_ticket_statuses` — the statuses this ticket's board really offers.
- `search_hudu` / `get_hudu_article` / `get_hudu_asset` — the client's Hudu
  documentation: knowledge-base articles, documented assets and their fields,
  and the company notes. See *Hudu setup* below. Only offered when Hudu is
  configured.

**Actions it proposes** — the tech gets an editable card in the pod and nothing
reaches ConnectWise until they press the button:

| Ask | What appears |
| --- | --- |
| "note this internally" | Internal Analysis note, editable |
| "put that on the ticket for the client" | Discussion note, editable |
| "draft an email and send it" | Email to the contact, editable, confirm to send |
| "put this in progress" | Status picker for this board |
| "log 30 minutes" | The Add Time sheet, pre-filled |
| **Create KB** button, or "write this up in Hudu" | A Hudu KB article draft, if no article covers the fix yet |

The model can only ever *propose* a write. `/action` performs it, carrying
whatever the technician actually approved, attributed to them.

## Add Time and Resolve

**Add Time** opens with the work note already drafted from the chat (or, if
there was no chat, from the ticket's own notes and work log) — editable, with a
**Redraft** button. It also offers an optional customer update email, drafted
and editable, sent only if the tech ticks the box. The work note is posted to
Internal Analysis alongside the time entry unless that box is unticked.

**Resolve** is unchanged in shape: it drafts the internal resolution note
(technician-only) and the customer-facing Resolution, both editable, then logs
time and moves the ticket.

**Who gets the credit.** The pod guesses the working technician (the member
ConnectWise passes in the pod URL, else the ticket owner, else the first
resource). That guess can be wrong — a tech closing a colleague's ticket — so
before logging time, resolving, or sending a customer email the pod shows a
"logged under" check with a technician picker. The tech confirms the name or
changes it; the chosen member is remembered for the rest of the session.

## Pod size

The pod's on-screen height is controlled entirely by ConnectWise — the iframe
cannot request more room (the Hosted API postMessage protocol has no resize
message). To make the pod taller:

1. In ConnectWise: **System > Setup Tables > Manage Hosted API** > open the
   Hercules record.
2. Set **Pod Height** to the pixel height you want (it renders at exactly this
   height for everyone). ~500 is the generic vendor default; **800–900 suits a
   chat pod** — if the ticket screen's column gets longer than the window, the
   page scrolls, which beats scrolling a cramped chat. Tune to taste.
3. Save; techs reload the ticket screen.

Two escape hatches when even that is too small:

- The pod's top bar has a **pop-out button** (only shown while embedded) that
  opens the same pod full-screen in a browser tab.
- A second Manage Hosted API record with **Type = Tab** (same URL) renders
  Hercules as a full tab on the ticket screen at nearly full window height.

## ScreenConnect setup

Install the **RESTful API Manager** extension in ScreenConnect, then set:

- `RESTfulAuthenticationSecret`: a new, long random string
- `RESTfulUserName`: `Hercules`
- `RESTfulAllowedOrigin`: the public origin of this Hercules service, such as
  `https://your-hercules-service.up.railway.app`

Add matching values to the Hercules service environment:

```env
SCREENCONNECT_BASE_URL=https://your-instance.screenconnect.com
SCREENCONNECT_API_SECRET=the-value-from-RESTfulAuthenticationSecret
SCREENCONNECT_ORIGIN=https://your-hercules-service.up.railway.app
```

The pod resolves each ConnectWise configuration attached to the current ticket
by computer name. **ScreenConnect** launches normal control. **Backstage** opens
ScreenConnect's Join with Options dialog, where the technician can select the
Backstage logon session.

The Manage API member needs inquire access to Companies > Configurations and
Service Desk > Service Tickets. ScreenConnect technicians still authenticate
normally and must have permission to view and join the matched Access session.

## Hudu setup

Create an API key in Hudu (**Admin > API Keys**, read access is enough) and set:

```env
HUDU_BASE_URL=https://your-instance.huducloud.com
HUDU_API_KEY=the-key
```

Hercules then searches Hudu on the tech's behalf: articles, assets (with the
fields recorded on them, such as VPN or firewall details) and the client's
company notes. Results are scoped to the ticket's client by default; the
assistant can widen to NTM-wide articles or all clients when asked. Passwords
are never fetched — the assistant points the tech at Hudu for those.

**Capturing fixes.** The **Create KB** button in the pod asks Hercules to
write the fix up. It searches Hudu for an existing article first and links it
if there is one; otherwise it drafts an article in the style of the existing
KB (a purpose line, then short numbered steps) as a card with an editable
title and body. From there the tech can edit the draft directly, or ask
Hercules for changes in chat — each message carries the draft as it currently
stands, and the revised proposal replaces the card. Nothing is written until
the tech presses **Create article**. Passwords never go in an article.

**Company matching.** Hudu's ConnectWise integration stamps each Hudu company
with its ConnectWise company id, and Hercules resolves the ticket's company
through that (falling back to an exact name match). The answer is stored in
the `hudu_companies` table and in memory, so it is looked up once per client,
not on every chat turn; the pod also pre-resolves it when a ticket opens. A
"not in Hudu" answer is retried after six hours in case the client gets synced
later. To force a re-check, delete that client's row from `hudu_companies`.

## Admin dashboard settings

The Hercules admin dashboard (served by the customer-facing Hercules service at
`/admin`) can change two things about this pod at runtime, through the shared
Postgres table `hercules_admin_settings` (`app = 'internal'`):

- `default_model` — the model the pod's dropdown pre-selects (and always allows).
- `prompt_role` / `prompt_guidelines` — the editable parts of the system prompt.
  Ticket data, untrusted-data fences, the tool list and the computed rules in
  `build_system_prompt` stay code-owned.

`admin_settings.py` caches the rows for 30 s and publishes the built-in defaults as
`default.<key>` rows on boot, so the dashboard can diff and reset. No DB, or a DB
error, simply means built-in defaults.
