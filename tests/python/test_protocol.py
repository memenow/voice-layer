"""Tests for JSON-RPC 2.0 envelope builders."""

from __future__ import annotations

import unittest

from voicelayer_worker.protocol import JSONRPC_VERSION, make_error, make_result


class ProtocolEnvelopeTest(unittest.TestCase):
    def test_make_result_preserves_ids_and_omits_error(self) -> None:
        for req_id in (1, "req-1", None):
            envelope = make_result(req_id, {"status": "ok"})
            self.assertEqual(envelope["jsonrpc"], JSONRPC_VERSION)
            self.assertEqual(envelope["id"], req_id)
            self.assertEqual(envelope["result"], {"status": "ok"})
            self.assertNotIn("error", envelope)

    def test_make_error_handles_optional_data(self) -> None:
        plain = make_error(7, -32601, "Method not found")
        self.assertEqual(plain["jsonrpc"], JSONRPC_VERSION)
        self.assertEqual(plain["id"], 7)
        self.assertEqual(plain["error"], {"code": -32601, "message": "Method not found"})
        self.assertNotIn("result", plain)

        with_data = make_error("abc", -32004, "Unavailable", {"method": "compose"})
        self.assertEqual(with_data["error"]["data"], {"method": "compose"})

        empty_data = make_error(None, -32700, "Parse error", {})
        self.assertNotIn("data", empty_data["error"])
