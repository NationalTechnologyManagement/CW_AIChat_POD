"""The confirmed-write endpoints: /action, /add-time, /statuses.

These are the paths that actually change ConnectWise, so the mocks assert on
what would be sent, not just on the response shape.
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("POD_SECRET", "test-secret")

from fastapi.testclient import TestClient  # noqa: E402

import cw_client  # noqa: E402
import main  # noqa: E402

AUTH = {"X-Pod-Token": "test-secret"}

TICKET = {
    "id": 500, "summary": "VPN drops", "board": "Support Tier1", "board_id": 34,
    "status": "New", "company_id": 7, "company_name": "Acme",
    "contact_id": 9, "contact_name": "Dana Reed", "contact_email": "dana@acme.test",
    "owner_identifier": "owner.tech",
}

STATUSES = [
    {"id": 626, "name": "In Progress", "inactive": False, "closed": False, "no_time_entry": False},
    {"id": 850, "name": "Re-Opened", "inactive": False, "closed": False, "no_time_entry": True},
    {"id": 739, "name": "AUTOMATED STATUS BELOW (DO NOT USE)", "inactive": False,
     "closed": False, "no_time_entry": True},
]


class ActionEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.ticket = mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=TICKET))
        self.ticket.start()
        self.addCleanup(self.ticket.stop)

    def post(self, payload):
        return self.client.post("/action", json=payload, headers=AUTH)

    def test_internal_note_is_written_as_the_confirming_tech(self):
        note = mock.AsyncMock(return_value={"id": 1})
        with mock.patch.object(cw_client, "create_ticket_note", note):
            response = self.post({"ticket_id": 500, "action": "add_internal_note",
                                  "text": "Rebuilt the tunnel.", "member_identifier": "dana.tech"})
        self.assertTrue(response.json()["success"])
        kwargs = note.await_args.kwargs
        self.assertEqual(kwargs["member_identifier"], "dana.tech")
        self.assertTrue(kwargs["internal"])

    def test_discussion_note_lands_on_the_discussion_tab(self):
        note = mock.AsyncMock(return_value={"id": 1})
        with mock.patch.object(cw_client, "create_ticket_note", note):
            self.post({"ticket_id": 500, "action": "add_discussion_note", "text": "Update for you"})
        kwargs = note.await_args.kwargs
        # A note with no flag at all is filed nowhere a tech would look.
        self.assertFalse(kwargs["internal"])
        self.assertTrue(kwargs["detail"])
        self.assertEqual(kwargs["member_identifier"], "owner.tech")  # falls back to the ticket owner

    def test_customer_email_uses_the_time_entry_path(self):
        send = mock.AsyncMock(return_value={"id": 2})
        with mock.patch.object(cw_client, "send_email_to_contact", send):
            response = self.post({"ticket_id": 500, "action": "send_customer_email", "text": "Hi Dana"})
        self.assertIn("Dana Reed", response.json()["message"])
        self.assertEqual(send.await_args.kwargs["text"], "Hi Dana")

    def test_status_change_resolves_the_name_against_the_board(self):
        setter = mock.AsyncMock(return_value={})
        with mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=STATUSES)), \
             mock.patch.object(cw_client, "set_ticket_status", setter):
            response = self.post({"ticket_id": 500, "action": "set_ticket_status",
                                  "status_name": "in progress"})
        self.assertTrue(response.json()["success"])
        self.assertEqual(setter.await_args.args[1], 626)

    def test_an_unknown_status_reports_the_real_options_and_writes_nothing(self):
        setter = mock.AsyncMock()
        with mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=STATUSES)), \
             mock.patch.object(cw_client, "set_ticket_status", setter):
            response = self.post({"ticket_id": 500, "action": "set_ticket_status",
                                  "status_name": "Working Issue Now"})
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["statuses"], ["In Progress", "Re-Opened"])  # no DO NOT USE
        setter.assert_not_awaited()

    def test_empty_text_is_rejected_before_touching_connectwise(self):
        note = mock.AsyncMock()
        with mock.patch.object(cw_client, "create_ticket_note", note):
            response = self.post({"ticket_id": 500, "action": "add_internal_note", "text": "   "})
        self.assertEqual(response.status_code, 400)
        note.assert_not_awaited()

    def test_unknown_actions_are_refused(self):
        for action in ("log_time", "delete_ticket", ""):
            self.assertEqual(self.post({"ticket_id": 500, "action": action, "text": "x"}).status_code, 400)

    def test_a_connectwise_failure_is_reported_as_a_failure(self):
        boom = mock.AsyncMock(side_effect=cw_client.CWAPIError(400, "bad note"))
        with mock.patch.object(cw_client, "create_ticket_note", boom):
            response = self.post({"ticket_id": 500, "action": "add_internal_note", "text": "x"})
        self.assertEqual(response.status_code, 500)
        self.assertFalse(response.json()["success"])
        self.assertIn("bad note", response.json()["error"])

    def test_the_endpoint_still_needs_the_pod_token(self):
        self.assertEqual(
            self.client.post("/action", json={"ticket_id": 500, "action": "add_internal_note", "text": "x"}).status_code,
            403,
        )


class AddTimeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    BASE = {
        "ticket_id": 500,
        "time_start": "2026-07-31T09:00:00Z",
        "time_end": "2026-07-31T09:30:00Z",
        "notes": "Rebuilt the Outlook profile.",
        "member_identifier": "dana.tech",
    }

    def post(self, **overrides):
        return self.client.post("/add-time", json={**self.BASE, **overrides}, headers=AUTH)

    def test_work_notes_also_post_to_internal_analysis(self):
        entry = mock.AsyncMock(return_value={"id": 9, "actualHours": 0.5})
        with mock.patch.object(cw_client, "create_time_entry", entry):
            response = self.post()
        self.assertTrue(response.json()["success"])
        self.assertTrue(entry.await_args.kwargs["add_to_internal"])

    def test_the_internal_flag_can_be_turned_off(self):
        entry = mock.AsyncMock(return_value={"id": 9, "actualHours": 0.5})
        with mock.patch.object(cw_client, "create_time_entry", entry):
            self.post(add_to_internal=False)
        self.assertFalse(entry.await_args.kwargs["add_to_internal"])

    def test_the_customer_update_is_emailed_when_asked(self):
        entry = mock.AsyncMock(return_value={"id": 9, "actualHours": 0.5})
        send = mock.AsyncMock(return_value={"id": 10})
        with mock.patch.object(cw_client, "create_time_entry", entry), \
             mock.patch.object(cw_client, "send_email_to_contact", send):
            body = self.post(send_email=True, email_text="Hi Dana, quick update").json()
        self.assertTrue(body["email_sent"])
        self.assertEqual(send.await_args.kwargs["text"], "Hi Dana, quick update")
        self.assertEqual(send.await_args.kwargs["member_identifier"], "dana.tech")

    def test_no_email_is_sent_without_the_flag_or_the_text(self):
        entry = mock.AsyncMock(return_value={"id": 9, "actualHours": 0.5})
        send = mock.AsyncMock()
        with mock.patch.object(cw_client, "create_time_entry", entry), \
             mock.patch.object(cw_client, "send_email_to_contact", send):
            self.post(email_text="drafted but not sent")
            self.post(send_email=True, email_text="   ")
        send.assert_not_awaited()

    def test_a_failed_email_still_reports_the_logged_time(self):
        entry = mock.AsyncMock(return_value={"id": 9, "actualHours": 0.5})
        send = mock.AsyncMock(side_effect=cw_client.CWAPIError(500, "mail down"))
        with mock.patch.object(cw_client, "create_time_entry", entry), \
             mock.patch.object(cw_client, "send_email_to_contact", send):
            body = self.post(send_email=True, email_text="Hi Dana").json()
        self.assertTrue(body["success"])
        self.assertFalse(body["email_sent"])
        self.assertIn("email was not sent", body["warning"])

    def test_a_failed_time_entry_never_emails_the_customer(self):
        entry = mock.AsyncMock(side_effect=cw_client.CWAPIError(400, "bad time"))
        send = mock.AsyncMock()
        with mock.patch.object(cw_client, "create_time_entry", entry), \
             mock.patch.object(cw_client, "send_email_to_contact", send):
            response = self.post(send_email=True, email_text="Hi Dana")
        self.assertEqual(response.status_code, 500)
        self.assertFalse(response.json()["success"])
        send.assert_not_awaited()


class StatusesEndpointTests(unittest.TestCase):
    def test_only_usable_statuses_reach_the_picker(self):
        client = TestClient(main.app)
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=TICKET)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=STATUSES)):
            body = client.get("/statuses?ticketId=500", headers=AUTH).json()
        self.assertEqual(body["statuses"], ["In Progress", "Re-Opened"])
        self.assertEqual(body["current"], "New")


if __name__ == "__main__":
    unittest.main()
