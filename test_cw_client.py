import unittest

import httpx

import cw_client


class ConnectWiseConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        await cw_client.close_client()

    async def test_hydrates_id_only_ticket_configuration_reference(self):
        async def handler(request: httpx.Request):
            if request.url.path == "/service/tickets/86675/configurations":
                return httpx.Response(200, json=[{"id": 10430, "name": ""}])
            if request.url.path == "/company/configurations/10430":
                return httpx.Response(200, json={
                    "id": 10430,
                    "name": "NTM-3527",
                    "serialNumber": "SD01255H",
                })
            return httpx.Response(404)

        cw_client._client = httpx.AsyncClient(
            base_url="https://manage.example.com",
            transport=httpx.MockTransport(handler),
        )

        configurations = await cw_client.get_ticket_configurations(86675)

        self.assertEqual(configurations, [{"id": 10430, "name": "NTM-3527"}])

    async def test_uses_name_already_present_on_reference(self):
        request_paths = []

        async def handler(request: httpx.Request):
            request_paths.append(request.url.path)
            return httpx.Response(200, json=[{"id": 12, "name": "PC-01"}])

        cw_client._client = httpx.AsyncClient(
            base_url="https://manage.example.com",
            transport=httpx.MockTransport(handler),
        )

        configurations = await cw_client.get_ticket_configurations(7)

        self.assertEqual(configurations, [{"id": 12, "name": "PC-01"}])
        self.assertEqual(request_paths, ["/service/tickets/7/configurations"])


class FullTicketHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        await cw_client.close_client()

    def _mock(self, handler):
        cw_client._client = httpx.AsyncClient(
            base_url="https://manage.example.com",
            transport=httpx.MockTransport(handler),
        )

    async def test_notes_fetches_every_page(self):
        pages_requested = []

        async def handler(request: httpx.Request):
            page = int(request.url.params.get("page", "1"))
            pages_requested.append(page)
            size = int(request.url.params["pageSize"])
            if page == 1:
                notes = [{"id": i, "text": f"n{i}", "member": {"name": "T"}} for i in range(size)]
            elif page == 2:
                notes = [{"id": 900, "text": "last", "member": {"name": "T"}}]
            else:
                notes = []
            return httpx.Response(200, json=notes)

        self._mock(handler)
        notes = await cw_client.get_ticket_notes(1)
        self.assertEqual(pages_requested, [1, 2])
        self.assertEqual(len(notes), 251)
        self.assertEqual(notes[-1]["text"], "last")

    async def test_notes_limit_is_a_single_cheap_page(self):
        calls = []

        async def handler(request: httpx.Request):
            calls.append(dict(request.url.params))
            return httpx.Response(200, json=[{"id": 1, "text": "x", "member": {"name": "T"}}])

        self._mock(handler)
        await cw_client.get_ticket_notes(1, limit=10)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["pageSize"], "10")

    async def test_all_pages_stops_when_page_param_ignored(self):
        async def handler(request: httpx.Request):
            # Same full page regardless of ?page= — a paging-blind endpoint.
            return httpx.Response(200, json=[{"id": i} for i in range(250)])

        self._mock(handler)
        items = await cw_client._get_all_pages("/service/tickets/1/notes", {"pageSize": 250})
        self.assertEqual(len(items), 250)

    async def test_audit_trail_normalizes_and_sorts_newest_first(self):
        async def handler(request: httpx.Request):
            self.assertEqual(request.url.path, "/system/audittrail")
            self.assertEqual(request.url.params["type"], "Ticket")
            self.assertEqual(request.url.params["id"], "42")
            return httpx.Response(200, json=[
                {"text": "old", "enteredDate": "2026-01-01T00:00:00Z", "enteredBy": "a", "auditType": "Record"},
                {"text": "new", "enteredDate": "2026-06-01T00:00:00Z", "enteredBy": "b", "auditType": "Status"},
                {"text": "  ", "enteredDate": "2026-07-01T00:00:00Z"},
            ])

        self._mock(handler)
        trail = await cw_client.get_ticket_audit_trail(42)
        self.assertEqual([a["text"] for a in trail], ["new", "old"])
        self.assertEqual(trail[0]["member"], "b")

    async def test_time_entries_query_and_shape(self):
        async def handler(request: httpx.Request):
            self.assertEqual(request.url.path, "/time/entries")
            self.assertIn('chargeToType="ServiceTicket" AND chargeToId=7', request.url.params["conditions"])
            return httpx.Response(200, json=[{
                "id": 5, "member": {"name": "Tech"}, "timeStart": "2026-07-01T09:00:00Z",
                "timeEnd": "2026-07-01T10:00:00Z", "actualHours": 1.0,
                "billableOption": "Billable", "notes": "did work",
                "internalNotes": "secret detail", "emailContactFlag": True,
            }])

        self._mock(handler)
        entries = await cw_client.get_ticket_time_entries(7)
        self.assertEqual(entries, [{
            "id": 5, "member": "Tech", "time_start": "2026-07-01T09:00:00Z",
            "time_end": "2026-07-01T10:00:00Z", "hours": 1.0, "billable": "Billable",
            "notes": "did work", "internal_notes": "secret detail", "email_sent": True,
        }])


if __name__ == "__main__":
    unittest.main()
