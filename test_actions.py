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


class TimeStatusTests(unittest.TestCase):
    """Board 49 refuses time on most statuses: /add-time moves the ticket to
    In Progress first, but never reopens a closed ticket."""

    STATUSES = [
        {"id": 874, "name": "New", "inactive": False, "closed": False, "no_time_entry": True},
        {"id": 878, "name": "In Progress", "inactive": False, "closed": False, "no_time_entry": False},
        {"id": 882, "name": "Closed", "inactive": False, "closed": True, "no_time_entry": True},
    ]
    BODY = {"ticket_id": 500, "time_start": "2026-07-31T09:00:00Z",
            "time_end": "2026-07-31T09:30:00Z", "notes": "Fixed it.", "member_identifier": "dana.tech"}

    def run_add_time(self, status):
        calls = []
        ticket = {"id": 500, "board_id": 49, "status": status}
        set_status = mock.AsyncMock(side_effect=lambda *a: calls.append("status") or {})
        entry = mock.AsyncMock(side_effect=lambda **k: calls.append("time") or {"id": 9, "actualHours": 0.5})
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=ticket)),              mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=self.STATUSES)),              mock.patch.object(cw_client, "set_ticket_status", set_status),              mock.patch.object(cw_client, "create_time_entry", entry):
            response = TestClient(main.app).post("/add-time", json=self.BODY, headers=AUTH)
        return response.json(), set_status, calls

    def test_a_blocking_status_moves_to_in_progress_before_the_time_entry(self):
        body, set_status, calls = self.run_add_time("New")
        self.assertTrue(body["success"])
        set_status.assert_awaited_once_with(500, 878)
        self.assertEqual(calls, ["status", "time"])
        self.assertEqual(body["status_moved_to"], "In Progress")

    def test_a_status_that_allows_time_is_left_alone(self):
        body, set_status, _ = self.run_add_time("In Progress")
        set_status.assert_not_awaited()
        self.assertIsNone(body["status_moved_to"])

    def test_a_closed_ticket_is_never_reopened_to_log_time(self):
        _, set_status, _ = self.run_add_time("Closed")
        set_status.assert_not_awaited()


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


BOARD_49_TICKET = {
    **TICKET, "id": 94945, "board": "Support Tier 1", "board_id": 49, "status": "In Progress",
}
BOARD_49_STATUSES = [
    {"id": 901, "name": "New*", "inactive": False, "closed": False, "no_time_entry": False},
    {"id": 626, "name": "In Progress", "inactive": False, "closed": False, "no_time_entry": False},
    {"id": 920, "name": "Ready to QC", "inactive": False, "closed": True, "no_time_entry": False},
    {"id": 903, "name": "Completed*", "inactive": False, "closed": True, "no_time_entry": False},
    {"id": 907, "name": "Client Contact 1*", "inactive": False, "closed": False, "no_time_entry": False},
    {"id": 913, "name": "DNU-AutoClosed", "inactive": False, "closed": True, "no_time_entry": True},
    {"id": 914, "name": "Resolved-Automation", "inactive": False, "closed": True, "no_time_entry": True},
]


class FinalizeResolveStatusTests(unittest.TestCase):
    """Regression for tickets 94076 / 94945: Resolve must actually move the ticket."""

    def test_resolve_on_support_tier_1_moves_the_ticket_to_ready_to_qc(self):
        client = TestClient(main.app)
        set_status = mock.AsyncMock(return_value={})
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=BOARD_49_TICKET)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=BOARD_49_STATUSES)), \
             mock.patch.object(cw_client, "create_ticket_note", mock.AsyncMock(return_value={"id": 1})), \
             mock.patch.object(cw_client, "set_ticket_status", set_status):
            body = client.post("/finalize-resolve", json={
                "ticket_id": 94945,
                "internal_note": "Rebooted, cleared the stuck Outlook add-in.",
                "email_text": "All set — the laptop is responsive again.",
                "send_email": False,
                "member_identifier": "JMiranda",
            }, headers=AUTH).json()

        self.assertTrue(body["success"], body)
        self.assertTrue(body["status_set"], body)
        self.assertEqual(body["status_name"], "Ready to QC")
        self.assertEqual(body["warnings"], [])
        self.assertIn("ticket resolved", body["message"])
        set_status.assert_awaited_once_with(94945, 920)

    def test_ready_to_qc_is_not_preferred_on_other_boards(self):
        client = TestClient(main.app)
        ticket = {**TICKET, "board_id": 34}
        statuses = [
            {"id": 920, "name": "Ready to QC", "inactive": False, "closed": True},
            {"id": 903, "name": "Resolved", "inactive": False, "closed": True},
        ]
        set_status = mock.AsyncMock(return_value={})
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=ticket)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=statuses)), \
             mock.patch.object(cw_client, "create_ticket_note", mock.AsyncMock(return_value={"id": 1})), \
             mock.patch.object(cw_client, "set_ticket_status", set_status):
            body = client.post("/finalize-resolve", json={
                "ticket_id": 500, "internal_note": "n", "email_text": "", "send_email": False,
                "member_identifier": "athorn",
            }, headers=AUTH).json()

        self.assertEqual(body["status_name"], "Resolved")
        set_status.assert_awaited_once_with(500, 903)

    # Monitoring (board 48) has neither Resolved nor Completed — its done status is a
    # plain "Closed". Three resolves there on 10-07/10-08 ended "status NOT changed"
    # and the techs closed the tickets by hand within a minute.
    MONITORING_TICKET = {**TICKET, "id": 95347, "board": "Monitoring", "board_id": 48, "status": "In Progress"}
    MONITORING_STATUSES = [
        {"id": 801, "name": n, "inactive": False, "closed": n.startswith("Closed") or n == "Cancelled", "no_time_entry": False}
        for n in ["New", "Re Opened", "Assigned", "Scheduled", "In Progress", "Waiting on Client",
                  "Waiting on Vendor/Parts", "Closed", "Closed - Merged", "Cancelled"]
    ]
    for i, s in enumerate(MONITORING_STATUSES):
        s["id"] = 801 + i   # "Closed" → 808

    def test_resolve_on_the_monitoring_board_falls_back_to_a_plain_closed(self):
        client = TestClient(main.app)
        set_status = mock.AsyncMock(return_value={})
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=self.MONITORING_TICKET)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=self.MONITORING_STATUSES)), \
             mock.patch.object(cw_client, "create_ticket_note", mock.AsyncMock(return_value={"id": 1})), \
             mock.patch.object(cw_client, "set_ticket_status", set_status):
            body = client.post("/finalize-resolve", json={
                "ticket_id": 95347, "internal_note": "Alert cleared.", "email_text": "", "send_email": False,
                "member_identifier": "athorn",
            }, headers=AUTH).json()

        self.assertTrue(body["status_set"], body)
        self.assertEqual(body["status_name"], "Closed")
        self.assertIn("ticket resolved", body["message"])
        set_status.assert_awaited_once_with(95347, 808)

    def test_closed_is_never_fuzzy_matched_to_closed_merged(self):
        client = TestClient(main.app)
        only_merged = [s for s in self.MONITORING_STATUSES if s["name"] != "Closed"]
        set_status = mock.AsyncMock(return_value={})
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=self.MONITORING_TICKET)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=only_merged)), \
             mock.patch.object(cw_client, "create_ticket_note", mock.AsyncMock(return_value={"id": 1})), \
             mock.patch.object(cw_client, "set_ticket_status", set_status):
            body = client.post("/finalize-resolve", json={
                "ticket_id": 95347, "internal_note": "n", "email_text": "", "send_email": False,
                "member_identifier": "athorn",
            }, headers=AUTH).json()

        self.assertFalse(body["status_set"])
        set_status.assert_not_awaited()

    # Auto-elevate tickets are typed on their origin board, then moved to Support
    # Tier 1. The pod page still offers the origin board's subtypes, so the resolve
    # used to send e.g. subtype 457 and ConnectWise answered "Service SubType 457 not
    # found for Service Board 49" (95337, 95355) — the status still moved, but the
    # banner went red. Now the category is validated against the CURRENT board.
    BOARD_49_OPTIONS = {
        "types": [{"id": 295, "name": "Change Management"}, {"id": 296, "name": "Remediation"}],
        "subtypes": {"295": [{"id": 496, "name": "User"}], "296": [{"id": 484, "name": "Credential Issue"}]},
        "items": {},
    }

    def _finalize_with_category(self, ticket, type_id, subtype_id, item_id=None):
        client = TestClient(main.app)
        set_status = mock.AsyncMock(return_value={})
        set_category = mock.AsyncMock(return_value={})
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=ticket)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=BOARD_49_STATUSES)), \
             mock.patch.object(cw_client, "create_ticket_note", mock.AsyncMock(return_value={"id": 1})), \
             mock.patch.object(cw_client, "update_ticket_category", set_category), \
             mock.patch.object(cw_client, "set_ticket_status", set_status), \
             mock.patch.object(main, "_board_options", mock.AsyncMock(return_value=self.BOARD_49_OPTIONS)):
            body = client.post("/finalize-resolve", json={
                "ticket_id": ticket["id"], "internal_note": "n", "email_text": "", "send_email": False,
                "member_identifier": "athorn", "type_id": type_id, "subtype_id": subtype_id, "item_id": item_id,
            }, headers=AUTH).json()
        return body, set_category, set_status

    def test_a_subtype_from_the_old_board_is_dropped_with_a_note_and_the_resolve_still_completes(self):
        ticket = {**BOARD_49_TICKET, "type_id": None, "subtype_id": None, "item_id": None}
        body, set_category, set_status = self._finalize_with_category(ticket, type_id=295, subtype_id=457)

        self.assertTrue(body["success"], body)
        self.assertTrue(body["status_set"], body)
        self.assertEqual(body["status_name"], "Ready to QC")
        self.assertIn("ticket resolved", body["message"])
        self.assertEqual(len(body["warnings"]), 1)
        self.assertIn("Subtype 457", body["warnings"][0])
        self.assertIn("Support Tier 1", body["warnings"][0])
        set_category.assert_awaited_once_with(94945, type_id=295, subtype_id=None, item_id=None)
        set_status.assert_awaited_once_with(94945, 920)

    def test_a_valid_category_is_written_as_given(self):
        ticket = {**BOARD_49_TICKET, "type_id": None, "subtype_id": None, "item_id": None}
        body, set_category, _ = self._finalize_with_category(ticket, type_id=295, subtype_id=496)
        self.assertEqual(body["warnings"], [])
        self.assertTrue(body["category_set"])
        set_category.assert_awaited_once_with(94945, type_id=295, subtype_id=496, item_id=None)

    def test_a_category_the_ticket_already_has_is_not_rewritten(self):
        ticket = {**BOARD_49_TICKET, "type_id": 295, "subtype_id": 496, "item_id": None}
        body, set_category, _ = self._finalize_with_category(ticket, type_id=295, subtype_id=496)
        self.assertEqual(body["warnings"], [])
        self.assertFalse(body["category_set"])
        set_category.assert_not_awaited()

    # A board nobody configured, with none of Resolved / Completed / Closed but a
    # plain closed-type status of its own: Resolve falls back to it, never to a
    # merge / cancel / automation state, and names it in the response.
    def test_an_unconfigured_board_resolves_to_its_own_plain_closed_type_status(self):
        client = TestClient(main.app)
        statuses = [
            {"id": 701, "name": "New", "inactive": False, "closed": False, "no_time_entry": False},
            {"id": 702, "name": "In Progress", "inactive": False, "closed": False, "no_time_entry": False},
            {"id": 703, "name": "Done", "inactive": False, "closed": True, "no_time_entry": True},
            {"id": 704, "name": "Closed - Merged", "inactive": False, "closed": True, "no_time_entry": True},
            {"id": 705, "name": "Cancelled", "inactive": False, "closed": True, "no_time_entry": True},
        ]
        ticket = {**TICKET, "id": 777, "board": "Projects", "board_id": 77, "status": "In Progress"}
        set_status = mock.AsyncMock(return_value={})
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=ticket)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=statuses)), \
             mock.patch.object(cw_client, "create_ticket_note", mock.AsyncMock(return_value={"id": 1})), \
             mock.patch.object(cw_client, "set_ticket_status", set_status):
            body = client.post("/finalize-resolve", json={
                "ticket_id": 777, "internal_note": "n", "email_text": "", "send_email": False,
                "member_identifier": "athorn",
            }, headers=AUTH).json()
        self.assertTrue(body["status_set"], body)
        self.assertEqual(body["status_name"], "Done")
        set_status.assert_awaited_once_with(777, 703)

    def test_a_per_board_override_wins_over_the_generic_list(self):
        client = TestClient(main.app)
        set_status = mock.AsyncMock(return_value={})
        with mock.patch.dict(main.RESOLVE_STATUS_BY_BOARD, {49: "In Progress"}, clear=False), \
             mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=BOARD_49_TICKET)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=BOARD_49_STATUSES)), \
             mock.patch.object(cw_client, "create_ticket_note", mock.AsyncMock(return_value={"id": 1})), \
             mock.patch.object(cw_client, "set_ticket_status", set_status):
            body = client.post("/finalize-resolve", json={
                "ticket_id": 94945, "internal_note": "n", "email_text": "", "send_email": False,
                "member_identifier": "athorn",
            }, headers=AUTH).json()
        self.assertEqual(body["status_name"], "In Progress")
        set_status.assert_awaited_once_with(94945, 626)

    # Attribution: auto-elevate tickets are owned by "Automation". With no tech
    # selected the pod used to fall back to the owner and credit the time and the
    # resolution to the automation account (95355). Now it refuses and says why.
    def test_resolving_as_the_automation_owner_is_refused_until_a_tech_is_picked(self):
        client = TestClient(main.app)
        ticket = {**BOARD_49_TICKET, "owner_identifier": "Automation"}
        note = mock.AsyncMock(return_value={"id": 1})
        set_status = mock.AsyncMock(return_value={})
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=ticket)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=BOARD_49_STATUSES)), \
             mock.patch.object(cw_client, "create_ticket_note", note), \
             mock.patch.object(cw_client, "set_ticket_status", set_status):
            r = client.post("/finalize-resolve", json={
                "ticket_id": 94945, "internal_note": "n", "email_text": "", "send_email": False,
            }, headers=AUTH)
            body = r.json()
            self.assertEqual(r.status_code, 400)
            self.assertFalse(body["success"])
            self.assertIn("Automation", body["error"])
            note.assert_not_awaited(); set_status.assert_not_awaited()

            ok = client.post("/finalize-resolve", json={
                "ticket_id": 94945, "internal_note": "n", "email_text": "", "send_email": False,
                "member_identifier": "athorn",
            }, headers=AUTH).json()
            self.assertTrue(ok["success"], ok)
            self.assertTrue(ok["status_set"])

    def test_add_time_as_an_automation_account_is_refused(self):
        client = TestClient(main.app)
        entry = mock.AsyncMock(return_value={"id": 1, "actualHours": 0.5})
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=BOARD_49_TICKET)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=BOARD_49_STATUSES)), \
             mock.patch.object(cw_client, "create_time_entry", entry):
            r = client.post("/add-time", json={
                "ticket_id": 94945, "notes": "worked", "member_identifier": "Automation",
                "time_start": "2026-10-08T13:00:00Z", "time_end": "2026-10-08T13:30:00Z",
            }, headers=AUTH)
        self.assertEqual(r.status_code, 400)
        self.assertIn("automation account", r.json()["error"])
        entry.assert_not_awaited()

    # A status that refuses time entries and is CLOSED is left alone (the pod must
    # not silently reopen a ticket) — but the technician gets a plain-English reason.
    def test_time_on_a_closed_status_explains_itself(self):
        client = TestClient(main.app)
        ticket = {**BOARD_49_TICKET, "status": "Completed*"}
        rejected = cw_client.CWAPIError(400, '{"code":"InvalidObject","message":"timeEntry object is invalid","errors":[{"code":"TimeEntryApi","message":"Please update the status of this ticket before entering time."}]}')
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=ticket)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=BOARD_49_STATUSES)), \
             mock.patch.object(cw_client, "create_time_entry", mock.AsyncMock(side_effect=rejected)), \
             mock.patch.object(cw_client, "set_ticket_status", mock.AsyncMock(return_value={})):
            body = client.post("/finalize-resolve", json={
                "ticket_id": 94945, "internal_note": "n", "email_text": "", "send_email": False,
                "member_identifier": "athorn",
                "time_start": "2026-10-08T13:00:00Z", "time_end": "2026-10-08T13:30:00Z",
            }, headers=AUTH).json()
        self.assertFalse(body["success"])
        self.assertIn("does not accept time entries", body["error"])
        self.assertIn("Completed*", body["error"])

    def test_a_board_without_a_done_status_still_says_so_loudly(self):
        client = TestClient(main.app)
        bare = [s for s in BOARD_49_STATUSES if s["name"] in ("New*", "In Progress")]
        set_status = mock.AsyncMock(return_value={})
        with mock.patch.object(cw_client, "get_ticket", mock.AsyncMock(return_value=BOARD_49_TICKET)), \
             mock.patch.object(cw_client, "get_board_statuses", mock.AsyncMock(return_value=bare)), \
             mock.patch.object(cw_client, "create_ticket_note", mock.AsyncMock(return_value={"id": 1})), \
             mock.patch.object(cw_client, "set_ticket_status", set_status):
            body = client.post("/finalize-resolve", json={
                "ticket_id": 94945, "internal_note": "n", "email_text": "", "send_email": False,
                "member_identifier": "JMiranda",
            }, headers=AUTH).json()

        self.assertFalse(body["status_set"])
        self.assertIn("status NOT changed", body["message"])
        self.assertTrue(any("status left unchanged" in w for w in body["warnings"]))
        set_status.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
