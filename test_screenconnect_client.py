import json
import unittest

import httpx

import screenconnect_client as screenconnect


SESSION_ID = "a507630e-80b5-ec92-bb5e-bcb48d00ff6b"


class ScreenConnectClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        await screenconnect.close_client()
        screenconnect._base_url = ""

    async def test_get_sessions_by_name_sends_secret_and_returns_exact_access_match(self):
        async def handler(request: httpx.Request):
            self.assertEqual(request.method, "POST")
            self.assertTrue(str(request.url).endswith("/Service.ashx/GetSessionsByName"))
            self.assertEqual(request.headers["CTRLAuthHeader"], "test-secret")
            self.assertEqual(request.headers["Origin"], "https://hercules.example.com")
            self.assertEqual(json.loads(request.content), ["PC-01"])
            return httpx.Response(200, json=[
                {
                    "SessionID": SESSION_ID,
                    "Name": "PC-01",
                    "SessionType": "Access",
                    "GuestConnectedCount": 1,
                },
                {
                    "SessionID": "2dfc5e8b-c786-4a9f-8f73-6a7fe4598f76",
                    "Name": "PC-010",
                    "SessionType": "Access",
                    "GuestConnectedCount": 1,
                },
            ])

        screenconnect._base_url = "https://control.example.com"
        screenconnect._client = httpx.AsyncClient(
            base_url=screenconnect._base_url,
            headers={
                "CTRLAuthHeader": "test-secret",
                "Origin": "https://hercules.example.com",
            },
            transport=httpx.MockTransport(handler),
        )

        sessions = await screenconnect.get_sessions_by_name("PC-01")

        self.assertEqual(sessions, [{
            "session_id": SESSION_ID,
            "name": "PC-01",
            "online": True,
        }])

    def test_build_launch_urls(self):
        screenconnect._base_url = "https://control.example.com"

        self.assertEqual(
            screenconnect.build_launch_url(SESSION_ID),
            f"https://control.example.com/Host#Access///{SESSION_ID}/Join",
        )
        self.assertEqual(
            screenconnect.build_launch_url(SESSION_ID, "backstage"),
            f"https://control.example.com/Host#Access///{SESSION_ID}/JoinWithOptions",
        )

    async def test_resolve_computers_deduplicates_session_ids(self):
        async def handler(_request: httpx.Request):
            return httpx.Response(200, json=[{
                "SessionID": SESSION_ID,
                "Name": "PC-01",
                "SessionType": "Access",
                "GuestConnectedCount": 0,
            }])

        screenconnect._base_url = "https://control.example.com"
        screenconnect._client = httpx.AsyncClient(
            base_url=screenconnect._base_url,
            transport=httpx.MockTransport(handler),
        )

        sessions = await screenconnect.resolve_computers(["PC-01", "PC-01"])

        self.assertEqual(len(sessions), 1)
        self.assertFalse(sessions[0]["online"])
        self.assertTrue(sessions[0]["backstage_url"].endswith("/JoinWithOptions"))

    async def test_missing_connection_count_has_unknown_online_state(self):
        async def handler(_request: httpx.Request):
            return httpx.Response(200, json=[{
                "SessionID": SESSION_ID,
                "Name": "NTM-3527",
                "SessionType": 2,
            }])

        screenconnect._base_url = "https://control.example.com"
        screenconnect._client = httpx.AsyncClient(
            base_url=screenconnect._base_url,
            transport=httpx.MockTransport(handler),
        )

        sessions = await screenconnect.get_sessions_by_name("NTM-3527")

        self.assertEqual(len(sessions), 1)
        self.assertIsNone(sessions[0]["online"])


if __name__ == "__main__":
    unittest.main()
