from __future__ import annotations

import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from voicelayer_orchestrator.protocol import (  # noqa: E402
    JSONRPC_VERSION,
    make_error,
    make_result,
)


class MakeResultEnvelopeTest(unittest.TestCase):
    """Pin the success envelope shape consumed by the Rust daemon's worker
    bridge. Drift here would break correlation between requests and
    responses on the JSON-RPC stdio channel.
    """

    def test_make_result_envelope_shape(self) -> None:
        envelope = make_result(7, {"text": "hi"})
        self.assertEqual(
            envelope,
            {"jsonrpc": JSONRPC_VERSION, "id": 7, "result": {"text": "hi"}},
        )
        self.assertNotIn("error", envelope)

    def test_make_result_preserves_string_id(self) -> None:
        # JSON-RPC allows string ids; the bridge correlates outgoing
        # requests by id and would lose the response if the worker
        # coerced the value.
        envelope = make_result("abc", {"ok": True})
        self.assertEqual(envelope["id"], "abc")

    def test_make_result_preserves_null_id(self) -> None:
        # A null id is the JSON-RPC default for cases where the caller
        # could not be correlated (e.g. parse errors). The factory must
        # pass it through unchanged.
        envelope = make_result(None, {})
        self.assertIsNone(envelope["id"])


class MakeErrorEnvelopeTest(unittest.TestCase):
    """Pin the error envelope shape and verify it carries the JSON-RPC
    server-error range fields the worker uses for transport diagnostics.
    """

    def test_make_error_envelope_shape(self) -> None:
        envelope = make_error(7, -32001, "boom")
        self.assertEqual(envelope["jsonrpc"], JSONRPC_VERSION)
        self.assertEqual(envelope["id"], 7)
        self.assertEqual(envelope["error"]["code"], -32001)
        self.assertEqual(envelope["error"]["message"], "boom")
        self.assertNotIn("data", envelope["error"])
        self.assertNotIn("result", envelope)

    def test_make_error_with_data(self) -> None:
        envelope = make_error(7, -32005, "provider failed", {"method": "transcribe"})
        self.assertEqual(envelope["error"]["data"], {"method": "transcribe"})

    def test_make_error_passes_nonstandard_code_through(self) -> None:
        # ``make_error`` is intentionally permissive about the numeric
        # value of ``code``: callers in :mod:`worker` use a small set of
        # constants but the factory itself does not validate the range,
        # so a forward-compatible extension can introduce new codes
        # without touching the protocol layer.
        envelope = make_error(7, 12345, "weird code")
        self.assertEqual(envelope["error"]["code"], 12345)

    def test_make_error_preserves_string_id(self) -> None:
        envelope = make_error("abc", -32600, "bad request")
        self.assertEqual(envelope["id"], "abc")

    def test_make_error_preserves_null_id(self) -> None:
        # Parse errors surface with id=null because the caller's id
        # could not be decoded; the factory must keep that semantic.
        envelope = make_error(None, -32700, "Unable to parse JSON input.")
        self.assertIsNone(envelope["id"])
        self.assertEqual(envelope["error"]["code"], -32700)

    def test_make_error_omits_data_when_empty(self) -> None:
        # The factory drops empty ``data`` dicts so the wire format
        # stays compact for the common case of "no extra detail".
        envelope = make_error(7, -32601, "nope", data={})
        self.assertNotIn("data", envelope["error"])


if __name__ == "__main__":
    unittest.main()
