"""Which ConnectWise tab does each pod write land on?

Discussion is the customer-visible tab. The rule (Gene, 2026-11-12): only a
customer email goes to Discussion; a technician's internal notes stay on
Internal Analysis. The bug this file pins: time entries carried the internal
note with the Internal flag set but the Discussion flag left out, and
ConnectWise ticks Discussion by default when it is omitted, so every internal
note logged with time was copied onto the customer-visible tab.

These tests assert on the JSON that would actually reach ConnectWise (a mock
HTTP transport under the real cw_client functions), not on mocked function
arguments, so a flag that is missing from the payload fails here.
"""

import json
import os
import unittest
from unittest import mock

import httpx

os.environ.setdefault("POD_SECRET", "test-secret")

from fastapi.testclient import TestClient  # noqa: E402

import cw_client  # noqa: E402
import main  # noqa: E402

AUTH = {"X-Pod-Token": "test-secret"}

TICKET = {
    "id": 94945, "summary": "Read access to shared documents", "board": "Support Tier 1",
    "board_id": 49, "status": "In Progress", "company_id": 7, "company_name": "Acme",
    "contact_id": 9, "contact_name": "Dana Reed", "contact_email": "dana@acme.test",
    "owner_identifier": "owner.tech",
}
STATUSES = [
    {"id": 878, "name": "In Progress", "inactive": False, "closed": False, "no_time_entry": False},
    {"id": 881, "name": "Ready to QC*", "inactive": False, "closed": False, "no_time_entry": False},
]

INTERNAL_NOTE = "[Resolution - CW Chat Pod]\n\nISSUE: read access requested.\nSTATUS: Resolved"
CUSTOMER_EMAIL = "Hi Dana, everyone now has read access to the shared documents."


class Recorder:
    """Stands in for ConnectWise and keeps every write the pod sends."""

    def __init__(self):
        self.writes = []  # (path, json body)

    async def handler(self, request: httpx.Request):
        if request.method == "POST":
            self.writes.append((request.url.path, json.loads(request.content)))
            return httpx.Response(201, json={"id": len(self.writes), "actualHours": 0.5})
        return httpx.Response(404)

    def install(self):
        cw_client._client = httpx.AsyncClient(
            base_url="https://manage.example.com",
            transport=httpx.MockTransport(self.handler),
        )

    def time_entries(self):
        return [body for path, body in self.writes if path == "/time/entries"]

    def ticket_notes(self):
        return [body for path, body in self.writes if path.endswith("/notes")]

    def discussion_writes(self):
        """Every write that would show on the customer-visible Discussion tab.

        A time entry that leaves addToDetailDescriptionFlag out counts: that is
        ConnectWise's default, and it is exactly how the bug happened.
        """
        hits = []
        for path, body in self.writes:
            if path == "/time/entries":
                if "notes" in body and body.get("addToDetailDescriptionFlag", True):
                    hits.append(body)
            elif body.get("detailDescriptionFlag", False):
                hits.append(body)
        return hits


class TimeEntryPayloadTests(unittest.IsolatedAsyncioTestCase):
    """cw_client.create_time_entry: the flags that reach ConnectWise."""

    async def asyncSetUp(self):
        self.cw = Recorder()
        self.cw.install()

    async def asyncTearDown(self):
        await cw_client.close_client()

    async def _entry(self, **kwargs):
        await cw_client.create_time_entry(
            ticket_id=500, time_start="2026-11-12T15:07:00Z", time_end="2026-11-12T15:37:00Z",
            member_identifier="galdaco", **kwargs)
        (body,) = self.cw.time_entries()
        return body

    async def test_internal_notes_go_to_internal_and_never_to_discussion(self):
        body = await self._entry(notes=INTERNAL_NOTE, add_to_internal=True)
        self.assertIs(body["addToDetailDescriptionFlag"], False)
        self.assertIs(body["addToInternalAnalysisFlag"], True)
        self.assertIs(body["addToResolutionFlag"], False)
        self.assertEqual(self.cw.discussion_writes(), [])

    async def test_notes_kept_off_the_ticket_still_never_reach_discussion(self):
        # Add Time with "Also post these notes to Internal Analysis" unticked:
        # the notes stay on the time entry. Before the fix they fell through to
        # ConnectWise's default and showed on Discussion instead.
        body = await self._entry(notes="Rebuilt the Outlook profile.", add_to_internal=False)
        self.assertIs(body["addToDetailDescriptionFlag"], False)
        self.assertIs(body["addToInternalAnalysisFlag"], False)
        self.assertIs(body["addToResolutionFlag"], False)
        self.assertEqual(self.cw.discussion_writes(), [])

    async def test_the_discussion_flag_is_explicit_even_without_notes(self):
        body = await self._entry()
        self.assertNotIn("notes", body)
        self.assertIs(body["addToDetailDescriptionFlag"], False)

    async def test_the_flag_is_a_real_json_false_not_merely_absent(self):
        await self._entry(notes=INTERNAL_NOTE, add_to_internal=True)
        (path, body), = self.cw.writes
        self.assertIn("addToDetailDescriptionFlag", body)


class EmailAndNotePayloadTests(unittest.IsolatedAsyncioTestCase):
    """The writes that are allowed on Discussion, and the note default."""

    async def asyncSetUp(self):
        self.cw = Recorder()
        self.cw.install()

    async def asyncTearDown(self):
        await cw_client.close_client()

    async def test_a_customer_email_is_what_goes_to_discussion(self):
        await cw_client.send_email_to_contact(500, CUSTOMER_EMAIL, member_identifier="galdaco")
        (body,) = self.cw.time_entries()
        self.assertIs(body["emailContactFlag"], True)
        self.assertIs(body["addToDetailDescriptionFlag"], True)
        self.assertIs(body["addToInternalAnalysisFlag"], False)
        self.assertEqual(self.cw.discussion_writes(), [body])

    async def test_a_resolution_email_goes_to_resolution_not_discussion(self):
        await cw_client.send_email_to_contact(500, CUSTOMER_EMAIL, member_identifier="galdaco", resolution=True)
        (body,) = self.cw.time_entries()
        self.assertIs(body["emailContactFlag"], True)
        self.assertIs(body["addToResolutionFlag"], True)
        self.assertIs(body["addToDetailDescriptionFlag"], False)
        self.assertIs(body["addToInternalAnalysisFlag"], False)

    async def test_a_ticket_note_is_internal_unless_asked_otherwise(self):
        await cw_client.create_ticket_note(500, "Checked the ACL on the share.", member_identifier="galdaco")
        (body,) = self.cw.ticket_notes()
        self.assertIs(body["internalAnalysisFlag"], True)
        self.assertIs(body["detailDescriptionFlag"], False)
        self.assertIs(body["resolutionFlag"], False)


class EndToEndVisibilityTests(unittest.TestCase):
    """The pod's endpoints, down to the JSON sent to ConnectWise."""

    def setUp(self):
        self.cw = Recorder()
        self.cw.install()
        self.addCleanup(setattr, cw_client, "_client", None)
        for name, value in (
            ("get_ticket", mock.AsyncMock(return_value=TICKET)),
            ("get_board_statuses", mock.AsyncMock(return_value=STATUSES)),
            ("set_ticket_status", mock.AsyncMock(return_value={})),
        ):
            patcher = mock.patch.object(cw_client, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)

    def resolve(self, **overrides):
        payload = {
            "ticket_id": 94945, "internal_note": INTERNAL_NOTE, "member_identifier": "galdaco",
            "time_start": "2026-11-12T15:07:00Z", "time_end": "2026-11-12T15:37:00Z",
            "send_email": True, "email_text": CUSTOMER_EMAIL,
        }
        payload.update(overrides)
        body = self.client.post("/finalize-resolve", json=payload, headers=AUTH).json()
        self.assertTrue(body["success"], body)
        return body

    def test_resolve_with_time_keeps_the_internal_note_off_discussion(self):
        # The exact case in Gene's screenshot: Resolve, time logged, email sent.
        body = self.resolve()
        self.assertTrue(body["time_logged"] and body["internal_note_saved"] and body["email_sent"], body)

        internal, email = self.cw.time_entries()
        self.assertEqual(internal["notes"], INTERNAL_NOTE)
        self.assertIs(internal["addToInternalAnalysisFlag"], True)
        self.assertIs(internal["addToDetailDescriptionFlag"], False)
        self.assertNotIn("emailContactFlag", internal)

        self.assertEqual(email["notes"], CUSTOMER_EMAIL)
        self.assertIs(email["emailContactFlag"], True)
        self.assertIs(email["addToResolutionFlag"], True)

        # Nothing in a resolve lands on Discussion: the internal note is
        # internal and the customer summary is the Resolution.
        self.assertEqual(self.cw.discussion_writes(), [])
        for write in self.cw.discussion_writes() + [email]:
            self.assertNotIn("ISSUE:", write["notes"])

    def test_resolve_with_time_and_no_email_writes_nothing_to_discussion(self):
        self.resolve(send_email=False)
        (internal,) = self.cw.time_entries()
        self.assertIs(internal["addToDetailDescriptionFlag"], False)
        (resolution,) = self.cw.ticket_notes()
        self.assertIs(resolution["resolutionFlag"], True)
        self.assertIs(resolution["detailDescriptionFlag"], False)
        self.assertEqual(self.cw.discussion_writes(), [])

    def test_resolve_without_time_writes_a_plain_internal_note(self):
        self.resolve(time_start=None, time_end=None, send_email=False, email_text="")
        self.assertEqual(self.cw.time_entries(), [])
        (note,) = self.cw.ticket_notes()
        self.assertIs(note["internalAnalysisFlag"], True)
        self.assertIs(note["detailDescriptionFlag"], False)
        self.assertEqual(self.cw.discussion_writes(), [])

    def add_time(self, **overrides):
        payload = {
            "ticket_id": 94945, "time_start": "2026-11-12T15:07:00Z", "time_end": "2026-11-12T15:37:00Z",
            "notes": "Rebuilt the Outlook profile.", "member_identifier": "galdaco",
        }
        payload.update(overrides)
        body = self.client.post("/add-time", json=payload, headers=AUTH).json()
        self.assertTrue(body["success"], body)
        return body

    def test_add_time_notes_stay_internal(self):
        self.add_time()
        (entry,) = self.cw.time_entries()
        self.assertIs(entry["addToInternalAnalysisFlag"], True)
        self.assertIs(entry["addToDetailDescriptionFlag"], False)
        self.assertEqual(self.cw.discussion_writes(), [])

    def test_add_time_with_the_internal_box_unticked_still_avoids_discussion(self):
        self.add_time(add_to_internal=False)
        (entry,) = self.cw.time_entries()
        self.assertIs(entry["addToInternalAnalysisFlag"], False)
        self.assertIs(entry["addToDetailDescriptionFlag"], False)
        self.assertEqual(self.cw.discussion_writes(), [])

    def test_add_time_with_a_customer_update_puts_only_the_email_on_discussion(self):
        body = self.add_time(send_email=True, email_text=CUSTOMER_EMAIL)
        self.assertTrue(body["email_sent"], body)
        (only,) = self.cw.discussion_writes()
        self.assertEqual(only["notes"], CUSTOMER_EMAIL)
        self.assertIs(only["emailContactFlag"], True)


if __name__ == "__main__":
    unittest.main()
