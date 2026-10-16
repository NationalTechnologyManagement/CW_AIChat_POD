"""The pod's "Text" button: /sms/info and /sms/send relay to ntm-sms-agent.

The agent owns the texting rules; these tests pin what the pod adds on top:
pod-token auth, a real technician on every send, and honest errors when the
agent is missing or down (Gene, 2026-10-16).
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("POD_SECRET", "test-secret")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import sms_agent_client  # noqa: E402

AUTH = {"X-Pod-Token": "test-secret"}

INFO = {
    "success": True, "ticketId": 500, "contactName": "Dana Reed", "poc": False, "canSend": True,
    "reason": None, "phone": "(248) 555-0101", "lineType": "mobile", "dryRun": False, "mode": "live",
    "history": [],
}


class SmsEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        patcher = mock.patch.object(sms_agent_client, "is_configured", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_info_is_relayed_from_the_agent(self):
        with mock.patch.object(sms_agent_client, "ticket_info", mock.AsyncMock(return_value=(200, INFO))) as info:
            r = self.client.get("/sms/info?ticketId=500", headers=AUTH)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["phone"], "(248) 555-0101")
        info.assert_awaited_once_with(500)

    def test_send_goes_out_as_the_confirming_tech(self):
        sent = {"success": True, "delivery": "sent", "to": "(248) 555-0101", "warning": None}
        with mock.patch.object(sms_agent_client, "send_text", mock.AsyncMock(return_value=(200, sent))) as send:
            r = self.client.post("/sms/send", json={"ticket_id": 500, "text": "On my way", "member_identifier": "tVaughan"}, headers=AUTH)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["delivery"], "sent")
        send.assert_awaited_once_with(500, "On my way", "tVaughan")

    def test_a_send_without_a_real_technician_is_refused(self):
        with mock.patch.object(sms_agent_client, "send_text", mock.AsyncMock()) as send:
            for member in (None, "", "Automation", "apimember"):
                r = self.client.post("/sms/send", json={"ticket_id": 500, "text": "hi", "member_identifier": member}, headers=AUTH)
                self.assertEqual(r.status_code, 400, member)
        send.assert_not_awaited()

    def test_agent_refusals_keep_their_status_and_message(self):
        refusal = {"success": False, "error": "This contact is a POC and has not enrolled in text updates."}
        with mock.patch.object(sms_agent_client, "send_text", mock.AsyncMock(return_value=(409, refusal))):
            r = self.client.post("/sms/send", json={"ticket_id": 500, "text": "hi", "member_identifier": "tVaughan"}, headers=AUTH)
        self.assertEqual(r.status_code, 409)
        self.assertIn("POC", r.json()["error"])

    def test_an_unreachable_agent_is_a_502_not_a_crash(self):
        boom = mock.AsyncMock(side_effect=httpx.ConnectError("refused"))
        with mock.patch.object(sms_agent_client, "ticket_info", boom):
            r = self.client.get("/sms/info?ticketId=500", headers=AUTH)
        self.assertEqual(r.status_code, 502)
        self.assertFalse(r.json()["success"])

    def test_not_configured(self):
        with mock.patch.object(sms_agent_client, "is_configured", return_value=False):
            r = self.client.post("/sms/send", json={"ticket_id": 500, "text": "hi", "member_identifier": "tVaughan"}, headers=AUTH)
        self.assertEqual(r.status_code, 503)

    def test_pod_token_required(self):
        self.assertEqual(self.client.get("/sms/info?ticketId=500").status_code, 403)
        self.assertEqual(self.client.post("/sms/send", json={"ticket_id": 500, "text": "x"}).status_code, 403)


class SmsAgentClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, json=INFO)
            return httpx.Response(409, json={"success": False, "error": "nope"})

        sms_agent_client._client = httpx.AsyncClient(
            base_url="https://sms.test", headers={"X-Tech-Secret": "tech"}, transport=httpx.MockTransport(handler))

    async def asyncTearDown(self):
        await sms_agent_client.close_client()

    async def test_calls_the_agent_with_the_tech_secret(self):
        status, body = await sms_agent_client.ticket_info(500)
        self.assertEqual((status, body["canSend"]), (200, True))
        status, body = await sms_agent_client.send_text(500, "hello", "tVaughan")
        self.assertEqual((status, body["error"]), (409, "nope"))
        self.assertEqual(self.requests[0].url.path, "/api/tickets/500/sms")
        self.assertEqual(self.requests[1].headers["X-Tech-Secret"], "tech")
        self.assertIn(b'"memberIdentifier":"tVaughan"', self.requests[1].content.replace(b" ", b""))

    def test_init_requires_url_and_secret(self):
        with mock.patch.dict(os.environ, {"SMS_AGENT_URL": "https://sms.test", "SMS_AGENT_SECRET": ""}):
            sms_agent_client.init_client()
        self.assertFalse(sms_agent_client.is_configured())


if __name__ == "__main__":
    unittest.main()
