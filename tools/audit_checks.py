"""Offline regression checks for confirmed bugs found during the code audit."""

import argparse
import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train


class InspectSingleRowTests(unittest.TestCase):
    def test_one_valid_row_reports_token_statistics(self):
        transformers_stub = types.ModuleType("transformers")

        class AutoTokenizerStub:
            @staticmethod
            def from_pretrained(_name):
                return object()

        transformers_stub.AutoTokenizer = AutoTokenizerStub
        args = argparse.Namespace(base="offline-stub", data="tiny.jsonl", seed=1, n=1)
        output = io.StringIO()

        with patch.dict(sys.modules, {"transformers": transformers_stub}):
            with patch.object(train, "triples", return_value=iter([("question", "thought", "solution")])):
                with patch.object(train, "encode", return_value=([1], [2, 3])):
                    with patch.object(train, "build_prompt", return_value="prompt"):
                        with patch.object(train, "build_response", return_value="response"):
                            with contextlib.redirect_stdout(output):
                                train.cmd_inspect(args)

        self.assertIn("Token lengths: median 3, p90 3, max 3", output.getvalue())


if __name__ == "__main__":
    unittest.main()
