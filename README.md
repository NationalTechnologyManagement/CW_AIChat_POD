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

## Secrets (Azure Key Vault)

API keys and shared secrets are not stored on the host. On boot, `vault.py`
reads them from the **kv-ntm-apps** Key Vault (resource group `rg-ntm-secrets`)
and writes each one into the process environment under the variable name the
code already uses, so every client module stays env-driven. The vault wins over
anything the platform has set.

| Environment variable       | Vault secret                  | Shared with                     |
| -------------------------- | ----------------------------- | ------------------------------- |
| `CW_CLIENT_ID`             | `cw-client-id`                | every NTM service (one key set) |
| `CW_PUBLIC_KEY`            | `cw-public-key`               | every NTM service (one key set) |
| `CW_PRIVATE_KEY`           | `cw-private-key`              | every NTM service (one key set) |
| `HUDU_API_KEY`             | `hudu-api-key`                | every NTM service               |
| `OPENROUTER_API_KEY`       | `openrouter-api-key`          |                                 |
| `SCREENCONNECT_API_SECRET` | `screenconnect-api-secret`    | ScreenConnect RESTful extension |
| `SMS_AGENT_SECRET`         | `sms-agent-tech-api-secret`   | ntm-sms-agent (`TECH_API_SECRET`) |
| `PIA_API_SECRET`           | `pia-api-secret`              | Pia API app ("full control")    |
| `POD_SECRET`               | `hercules-pod-secret`         | the Hercules pod clients        |
| `LIVE_BRIDGE_SECRET`       | `hercules-live-bridge-secret` | the Hercules Node server        |
| `CALL_BRIDGE_SECRET`       | `hercules-ingest-secret`      | the voice bridge                |

The service authenticates to the vault as the `hercules-tech-keyvault-reader`
service principal, which holds **Key Vault Secrets User** on the vault and
nothing else. Only these four variables stay on the platform (Railway):

```env
AZURE_KEY_VAULT_URL=https://kv-ntm-apps.vault.azure.net
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
```

The vault firewall denies by default and allows only the Hercules-Tech
service's three Railway **Static Outbound IPs** (Settings > Networking on the
service; region `iad`) plus admin workstations. If the service moves region or
static IPs are toggled, the new addresses must be added to the vault's
network rules before the next deploy, or boot fails with a 403 from the vault.

Non-secret configuration (`CW_API_URL`, `CW_COMPANY_ID`, `HUDU_BASE_URL`,
`SCREENCONNECT_BASE_URL`, `SMS_AGENT_URL`, `DATABASE_URL`, `REDIS_URL`, ...)
remains ordinary environment variables.

To rotate a key, set a new version of the secret in the vault and restart the
service; nothing on the platform changes. A vault that is configured but
unreachable or that refuses the credential is fatal on boot. A single secret
that is missing from the vault is logged and the environment value, if any, is
used instead.

Without `AZURE_KEY_VAULT_URL` (tests, local dev) the vault is skipped and `.env`
applies as before. A developer who is signed in with `az login` and has
**Key Vault Secrets User** can set just the URL and get the live keys locally.

## ScreenConnect setup

Install the **RESTful API Manager** extension in ScreenConnect, then set:

- `RESTfulAuthenticationSecret`: a new, long random string
- `RESTfulUserName`: `Hercules`
- `RESTfulAllowedOrigin`: the public origin of this Hercules service, such as
  `https://your-hercules-service.up.railway.app`

Store the secret in the vault as `screenconnect-api-secret` and add the
non-secret values to the Hercules service environment:

```env
SCREENCONNECT_BASE_URL=https://your-instance.screenconnect.com
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

Create an API key in Hudu (**Admin > API Keys**, read access is enough), store
it in the vault as `hudu-api-key`, and set:

```env
HUDU_BASE_URL=https://your-instance.huducloud.com
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

## Pia setup

Pia (pia.ai) runs NTM's service-desk automations — account unlocks, password
resets, mailbox and calendar permissions, restarts, Outlook/printer/OneDrive
troubleshooting, onboarding and offboarding. Techs normally reach them through
the separate Pia pod on the ticket; Hercules brings them into its own chat.

In the Pia tenant create an API app (**Users > Manage API Apps**, every scope —
the tools need `TechAssist.Read`, `Packages.Execute` and
`Packages.MonitorExecution`), store its secret in the vault as `pia-api-secret`,
and set:

```env
PIA_BASE_URL=https://trustntm.pia.ai
```

**Verified per client first.** Pia stamps each of its clients with the
ConnectWise company id (`psaCompanyId`) and publishes Tech Assist automations
per PSA company. When a ticket opens, Hercules looks the client up in Pia and
reads the automations published to it. A client that is not in Pia, or has
nothing published, gets no Pia tools, no Pia button and no mention of Pia —
the assistant is told not to offer it. Both lookups are cached for ten minutes.

**Reading the ticket.** For a Pia-enabled client the pod load also asks the
default model to match the ticket's summary and description against the
client's catalogue, strictly (same task, not same topic). Matches appear as a
*Pia can help* panel at the top of the chat with a **Start** button each; the
full catalogue is also in the assistant's system prompt, so in conversation it
can recognise "unlock Dana's account" as a Pia automation and offer it. A
**Pia** mode tab (beside Assistant, Live and Call, shown only for Pia-enabled
clients) holds Pia's own chat for the ticket at full pod size.

**Starting one** is a write and goes through the usual confirm card
(`run_pia_automation` → `/action`). The server re-checks that the client is
Pia-enabled and that the package is in *that client's* catalogue before
calling Pia. Pia then opens its own chat session on the ticket and asks the
tech for whatever it still needs (which computer, which user, which mailbox)
as a form with the choices filled in. The API exposes neither those choices
nor a way to submit the form, and the package input keys are not published,
so Hercules deliberately passes no inputs. Instead the card **frames Pia's own
chat for the ticket** (`/Extension/PiaEmbed/?ticketId=…`, the page Pia's
browser extension embeds into Autotask and Halo) as soon as the automation
starts, so the tech picks from Pia's list and submits without leaving
Hercules. That page is cookie-authenticated against the tech's Pia sign-in:
if it shows a sign-in page or stays blank, **Sign in to Pia** opens Pia's
`/Extension/Authenticate` in a popup (as the extension does) and the frame
reloads when it closes; **Open in Pia** is the same chat in a new tab. The
card also polls `/pia/status` and shows the state (queued, running, *needs your
answer below* with Pia's question, completed, failed), and the outcome is fed
back into the chat so `check_pia_automation` and follow-up questions are
grounded. Pia posts its own audit notes to the ticket as it runs.

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
