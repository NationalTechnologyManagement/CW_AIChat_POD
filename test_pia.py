"""Pia Tech Assist from the Hercules chat.

What these pin (Gene, 2026-10-24): Pia is verified for the ticket's CLIENT
before anything is offered; starting an automation is a confirm-card write that
the server re-validates against that client's catalogue; and Pia's "waiting for
form submission" state is surfaced as "answer it in the Pia pod", since the API
gives no way to fill Pia's form.
"""

import json
import os
import unittest
from unittest import mock

os.environ.setdefault("POD_SECRET", "test-secret")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import cw_tools  # noqa: E402
import main  # noqa: E402
import pia_client  # noqa: E402

AUTH = {"X-Pod-Token": "test-secret"}

TICKET = {"id": 500, "summary": "Unlock Dana's account", "company_id": 250, "company_name": "NTM",
          "board_id": 49, "contact_id": 7, "contact_email": "dana@example.com"}

CLIENTS_PAGE = {
    "next": None, "nextTop": 0, "totalRecords": 2, "count": 2,
    "value": [
        {"id": 31, "name": "National Technology Management", "psaCompanyId": "250", "isPiaChatbotEnabled": True, "tags": ["Cloud Only"]},
        {"id": 1, "name": "Heat Treating Services", "psaCompanyId": "19301", "isPiaChatbotEnabled": False, "tags": []},
    ],
}
UNLOCK = "dce4319b-8994-4677-80ab-db0ecbaa0e5d"
DETAILS = "0ebe7297-0f2e-45d6-ba71-e02a7de93258"
CATALOG_PAGE = {
    "next": None, "nextTop": 0, "totalRecords": 2, "count": 2,
    "value": [
        {"packageId": DETAILS, "optionButton": "Get staff account details", "optionText": "Help me find out information about a staff network account",
         "optionCategory": "Staff Changes", "automationName": "Pia: Get User Account Details (Cloud Only)", "description": ""},
        {"packageId": UNLOCK, "optionButton": "Unlock Account", "optionText": "Can you help unlock a user account?",
         "optionCategory": "Staff Changes", "automationName": "Pia: Account Locked Out (Cloud Only)", "description": ""},
    ],
}
WAITING_STATUS = {
    "eventCorrelationId": "2fbb41c5-b124-407f-b3d0-2f14d1afe402", "isProcessed": True,
    "packageExecutions": [{"packageInstanceId": 18, "packageName": "Pia: Get User Account Details (Cloud Only)",
                           "isCompleted": False, "isWaiting": True, "isCancelled": False, "isError": False,
                           "executionStart": "2026-10-24T19:00:40"}],
}
WAITING_LOGS = [
    {"logType": "Information", "activityStaticName": "extension_automation", "logMessage": "Start execute extension packages", "logDate": "2026-10-24T19:00:40"},
    {"logType": "Information", "activityStaticName": "chat_interaction", "logMessage": "New chat session created.", "logDate": "2026-10-24T19:00:41"},
    {"logType": "Verbose", "activityStaticName": "chat_interaction", "logMessage": "Message Text: Please select the user you need details for.", "logDate": "2026-10-24T19:00:41"},
    {"logType": "Information", "activityStaticName": "chat_interaction", "logMessage": "Message has form attached. Waiting for form submission.", "logDate": "2026-10-24T19:00:41"},
]


class FakePia:
    """pia_client talking to a canned Pia tenant through httpx.MockTransport."""

    def __init__(self, catalog_status=200):
        self.requests: list[httpx.Request] = []
        self.catalog_status = catalog_status

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/config/clients":
            return httpx.Response(200, json=CLIENTS_PAGE)
        if path.startswith("/api/automate/discover/techassist/psacompany/250"):
            return httpx.Response(self.catalog_status, json=CATALOG_PAGE if self.catalog_status == 200 else
                                  {"title": "Unable to retrieve data", "status": 500, "detail": "Something went wrong"})
        if path.startswith("/api/automate/discover/techassist/psacompany/"):
            return httpx.Response(500, json={"title": "Unable to retrieve data", "status": 500, "detail": "Something went wrong"})
        if path.startswith("/api/automate/trigger/"):
            return httpx.Response(200, json={"eventCorrelationId": "2fbb41c5-b124-407f-b3d0-2f14d1afe402"})
        if path.endswith("/status"):
            return httpx.Response(200, json=WAITING_STATUS)
        if path.endswith("/activitylogs"):
            return httpx.Response(200, json=WAITING_LOGS)
        return httpx.Response(404, json={"detail": path})

    def __enter__(self):
        pia_client._reset_caches()
        pia_client._client = httpx.AsyncClient(base_url="https://trustntm.pia.ai",
                                               headers={"Authorization": "api secret"},
                                               transport=httpx.MockTransport(self.handler))
        pia_client._base_url = "https://trustntm.pia.ai"
        return self

    def __exit__(self, *exc):
        pia_client._client = None
        pia_client._base_url = ""
        pia_client._reset_caches()


def run(coro):
    import asyncio
    return asyncio.run(coro)


class ReadinessTests(unittest.TestCase):
    def test_client_with_published_automations_is_ready(self):
        with FakePia():
            ready = run(pia_client.readiness(250))
        self.assertTrue(ready["enabled"])
        self.assertEqual(ready["reason"], "ready")
        self.assertEqual(ready["client"]["name"], "National Technology Management")
        self.assertEqual([a["button"] for a in ready["automations"]], ["Get staff account details", "Unlock Account"])

    def test_company_unknown_to_pia_is_not_ready(self):
        with FakePia() as fake:
            ready = run(pia_client.readiness(999999))
        self.assertFalse(ready["enabled"])
        self.assertEqual(ready["reason"], "not_in_pia")
        self.assertIn("not set up in Pia", pia_client.not_ready_message(ready, "Acme"))
        # Pia's 500 for an unknown company is "nothing published", not an error.
        self.assertTrue(any("psacompany/999999" in str(r.url) for r in fake.requests))

    def test_client_without_automations_is_not_ready(self):
        with FakePia(catalog_status=500):
            ready = run(pia_client.readiness(250))
        self.assertFalse(ready["enabled"])
        self.assertEqual(ready["reason"], "no_automations")

    def test_readiness_is_cached_per_company(self):
        with FakePia() as fake:
            run(pia_client.readiness(250))
            n = len(fake.requests)
            run(pia_client.readiness(250))
            self.assertEqual(len(fake.requests), n)
            self.assertTrue(pia_client.cached_readiness(250)["enabled"])
            self.assertEqual(pia_client.find_option(250, UNLOCK.upper())["button"], "Unlock Account")

    def test_not_configured(self):
        pia_client._client = None
        self.assertEqual(run(pia_client.readiness(250))["reason"], "not_configured")
        self.assertFalse(pia_client.is_configured())


class DescribeTests(unittest.TestCase):
    def test_waiting_on_form_names_pias_question(self):
        report = pia_client.describe(WAITING_STATUS, WAITING_LOGS)
        self.assertEqual(report["state"], "waiting_for_input")
        self.assertEqual(report["pia_question"], "Please select the user you need details for.")
        self.assertIn("Pia pod", report["summary"])

    def test_terminal_states(self):
        done = {**WAITING_STATUS, "packageExecutions": [{**WAITING_STATUS["packageExecutions"][0], "isWaiting": False, "isCompleted": True}]}
        self.assertEqual(pia_client.describe(done, [])["state"], "completed")
        failed = {**WAITING_STATUS, "packageExecutions": [{**WAITING_STATUS["packageExecutions"][0], "isError": True}]}
        report = pia_client.describe(failed, [{"logType": "Error", "activityStaticName": "x", "logMessage": "Graph said no"}])
        self.assertEqual(report["state"], "error")
        self.assertEqual(report["errors"], ["Graph said no"])
        self.assertEqual(pia_client.describe({"isProcessed": False, "packageExecutions": []}, [])["state"], "queued")
        self.assertEqual(pia_client.describe(None, [])["state"], "unknown")

    def test_trigger_rejects_non_guid_and_returns_correlation(self):
        with FakePia() as fake:
            with self.assertRaises(pia_client.PiaError):
                run(pia_client.trigger("not-a-guid", 500))
            corr = run(pia_client.trigger(UNLOCK, 500))
            self.assertEqual(corr, "2fbb41c5-b124-407f-b3d0-2f14d1afe402")
            req = fake.requests[-1]
            self.assertEqual(req.url.path, f"/api/automate/trigger/{UNLOCK}/withTicket/500")
            self.assertEqual(json.loads(req.content), {"variableInputs": []})
            self.assertEqual(req.headers["Authorization"], "api secret")


class ToolGatingTests(unittest.TestCase):
    def test_pia_tools_are_withheld_unless_the_client_is_verified(self):
        names = {s["function"]["name"] for s in cw_tools.specs_for(TICKET, pia_enabled=False)}
        self.assertFalse(names & cw_tools.PIA_TOOLS)
        names = {s["function"]["name"] for s in cw_tools.specs_for(TICKET, pia_enabled=True)}
        self.assertTrue(cw_tools.PIA_TOOLS <= names)
        self.assertIn("run_pia_automation", cw_tools.WRITE_TOOLS)
        self.assertTrue(cw_tools.PIA_READ_TOOLS <= cw_tools.READ_TOOLS)

    def test_default_gating_uses_the_cached_readiness(self):
        with FakePia():
            self.assertFalse({s["function"]["name"] for s in cw_tools.specs_for(TICKET)} & cw_tools.PIA_TOOLS)
            run(pia_client.readiness(250))
            self.assertTrue(cw_tools.PIA_TOOLS <= {s["function"]["name"] for s in cw_tools.specs_for(TICKET)})

    def test_proposal_card_names_the_automation_from_the_catalogue(self):
        with FakePia():
            run(pia_client.readiness(250))
            card = cw_tools.build_proposal("c1", "run_pia_automation", {"package_id": UNLOCK, "reason": "Dana is locked out."}, TICKET)
        self.assertEqual(card["action"], "run_pia_automation")
        self.assertEqual(card["package_id"], UNLOCK)
        self.assertEqual(card["automation_name"], "Pia: Account Locked Out (Cloud Only)")
        self.assertIn("Unlock Account", card["title"])
        self.assertEqual(card["confirm_label"], "Start in Pia")
        self.assertEqual(card["text"], "Dana is locked out.")
        receipt = json.loads(cw_tools.proposal_receipt("run_pia_automation", {}))
        self.assertEqual(receipt["status"], "awaiting_technician")

    def test_list_tool_reports_not_ready_plainly(self):
        with FakePia():
            payload = json.loads(run(cw_tools.run_read_tool("list_pia_automations", {}, {**TICKET, "company_id": 4242, "company_name": "Acme"}))
                                 .split("\n", 1)[1].rsplit("\n", 1)[0])
        self.assertFalse(payload["pia_enabled_for_client"])
        self.assertIn("not set up in Pia", payload["result"])

    def test_list_tool_returns_the_catalogue(self):
        with FakePia():
            text = run(cw_tools.run_read_tool("list_pia_automations", {}, TICKET))
        self.assertIn("UNTRUSTED PIA DATA", text)
        payload = json.loads(text.split("\n", 1)[1].rsplit("\n", 1)[0])
        self.assertTrue(payload["pia_enabled_for_client"])
        self.assertEqual(payload["automations"][1]["package_id"], UNLOCK)

    def test_prompt_section_lists_automations_or_forbids_pia(self):
        with FakePia():
            ready = run(pia_client.readiness(250))
            prompt = main.build_system_prompt(TICKET, [], tools_enabled=True, available_tools=["run_pia_automation"], pia=ready)
            self.assertIn("PIA AUTOMATIONS AVAILABLE FOR NTM", prompt)
            self.assertIn(UNLOCK, prompt)
            self.assertIn("Running Pia automations", prompt)
            off = run(pia_client.readiness(4242))
            prompt = main.build_system_prompt({**TICKET, "company_id": 4242, "company_name": "Acme"}, [], tools_enabled=True, available_tools=[], pia=off)
            self.assertIn("Do not offer or mention Pia", prompt)
            self.assertNotIn("run_pia_automation", prompt)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        patcher = mock.patch.object(main.cw_client, "get_ticket", mock.AsyncMock(return_value=TICKET))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_start_requires_a_package_from_this_clients_catalogue(self):
        with FakePia() as fake:
            r = self.client.post("/action", json={"ticket_id": 500, "action": "run_pia_automation",
                                                  "package_id": "11111111-1111-1111-1111-111111111111"}, headers=AUTH)
            self.assertEqual(r.status_code, 400)
            self.assertIn("not in this client's Pia catalogue", r.json()["error"])
            self.assertFalse(any("/trigger/" in str(req.url) for req in fake.requests))

    def test_start_refused_for_a_client_not_in_pia(self):
        with mock.patch.object(main.cw_client, "get_ticket", mock.AsyncMock(return_value={**TICKET, "company_id": 4242, "company_name": "Acme"})):
            with FakePia() as fake:
                r = self.client.post("/action", json={"ticket_id": 500, "action": "run_pia_automation", "package_id": UNLOCK}, headers=AUTH)
                self.assertEqual(r.status_code, 400)
                self.assertIn("not set up in Pia", r.json()["error"])
                self.assertFalse(any("/trigger/" in str(req.url) for req in fake.requests))

    def test_start_triggers_pia_against_the_ticket(self):
        with FakePia() as fake:
            r = self.client.post("/action", json={"ticket_id": 500, "action": "run_pia_automation", "package_id": UNLOCK}, headers=AUTH)
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertTrue(body["success"])
            self.assertEqual(body["correlation_id"], "2fbb41c5-b124-407f-b3d0-2f14d1afe402")
            self.assertIn("Unlock Account", body["message"])
            trig = [req for req in fake.requests if "/trigger/" in str(req.url)]
            self.assertEqual(trig[0].url.path, f"/api/automate/trigger/{UNLOCK}/withTicket/500")

    def test_status_endpoint_reports_waiting_with_pias_question(self):
        with FakePia():
            r = self.client.get("/pia/status?correlationId=2fbb41c5-b124-407f-b3d0-2f14d1afe402", headers=AUTH)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["state"], "waiting_for_input")
            self.assertEqual(r.json()["pia_question"], "Please select the user you need details for.")
            r = self.client.get("/pia/status?correlationId=nope", headers=AUTH)
            self.assertEqual(r.status_code, 400)
        # Pia not configured at all: the pod gets a clear 503, not a traceback.
        r = self.client.get("/pia/status?correlationId=2fbb41c5-b124-407f-b3d0-2f14d1afe402", headers=AUTH)
        self.assertEqual(r.status_code, 503)

    def test_everything_needs_the_pod_token(self):
        r = self.client.get("/pia/status?correlationId=2fbb41c5-b124-407f-b3d0-2f14d1afe402")
        self.assertEqual(r.status_code, 403)

    def test_pod_without_pia_configured_hides_it(self):
        pia_client._client = None
        pia = run(main._pia_for_pod(TICKET))
        self.assertFalse(pia["enabled"])
        self.assertEqual(pia["suggestions"], [])

    def test_pod_suggestions_come_from_reading_the_ticket(self):
        picks = [{"package_id": UNLOCK, "button": "Unlock Account", "name": "Pia: Account Locked Out (Cloud Only)", "reason": "Dana is locked out."}]
        with FakePia(), mock.patch.object(main.openrouter_client, "suggest_pia_automations", mock.AsyncMock(return_value=picks)) as suggest:
            pia = run(main._pia_for_pod(TICKET))
        self.assertTrue(pia["enabled"])
        self.assertEqual(pia["client_name"], "National Technology Management")
        self.assertEqual(pia["automation_count"], 2)
        self.assertEqual(pia["suggestions"], picks)
        suggest.assert_awaited_once()
        self.assertEqual(suggest.await_args.args[0]["id"], 500)

    def test_pod_for_unknown_client_asks_the_model_nothing(self):
        with FakePia(), mock.patch.object(main.openrouter_client, "suggest_pia_automations", mock.AsyncMock()) as suggest:
            pia = run(main._pia_for_pod({**TICKET, "company_id": 4242}))
        self.assertFalse(pia["enabled"])
        self.assertEqual(pia["reason"], "not_in_pia")
        suggest.assert_not_awaited()


class SuggestionParsingTests(unittest.TestCase):
    def test_model_picks_are_mapped_back_to_packages(self):
        import openrouter_client
        options = [pia_client._option_row(o) for o in CATALOG_PAGE["value"]]
        with mock.patch.object(openrouter_client, "_call_openrouter",
                               mock.AsyncMock(return_value='```json\n[{"n": 2, "reason": "Unlocks Dana."}, {"n": 9, "reason": "bogus"}, {"n": 2, "reason": "dup"}]\n```')):
            picks = run(openrouter_client.suggest_pia_automations(TICKET, options))
        self.assertEqual(picks, [{"package_id": UNLOCK, "button": "Unlock Account", "name": "Pia: Account Locked Out (Cloud Only)", "reason": "Unlocks Dana."}])
        with mock.patch.object(openrouter_client, "_call_openrouter", mock.AsyncMock(return_value="[]")):
            self.assertEqual(run(openrouter_client.suggest_pia_automations(TICKET, options)), [])
        with mock.patch.object(openrouter_client, "_call_openrouter", mock.AsyncMock(return_value="Sorry, nothing fits.")):
            self.assertEqual(run(openrouter_client.suggest_pia_automations(TICKET, options)), [])


if __name__ == "__main__":
    unittest.main()
