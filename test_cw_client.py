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


if __name__ == "__main__":
    unittest.main()
