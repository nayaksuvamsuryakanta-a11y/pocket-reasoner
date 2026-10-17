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

import grpo
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


class BatchArgumentTests(unittest.TestCase):
    def test_plan_steps_rejects_nonpositive_batch_factors(self):
        for batch_size, grad_accum in ((0, 1), (1, 0)):
            with self.subTest(batch_size=batch_size, grad_accum=grad_accum):
                with self.assertRaises(SystemExit):
                    train.plan_steps(8, batch_size, grad_accum, 1)

    def test_plan_batches_rejects_nonpositive_device_batch(self):
        with self.assertRaises(SystemExit):
            grpo.plan_batches(num_generations=8, per_device_batch=0, prompts_per_step=2)


if __name__ == "__main__":
    unittest.main()
