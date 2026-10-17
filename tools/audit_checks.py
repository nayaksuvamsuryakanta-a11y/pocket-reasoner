"""Offline regression checks for confirmed bugs found during the code audit."""

import argparse
import ast
import contextlib
import io
import itertools
import math
import statistics
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_functions(filename, names, namespace):
    path = PROJECT_ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"Could not find all requested functions in {filename}")
    scope = dict(namespace)
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


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
        scope = load_functions("train.py", {"cmd_inspect"}, {
            "itertools": itertools,
            "statistics": statistics,
            "DATASET": "offline dataset",
            "triples": lambda _data, _seed: iter([("question", "thought", "solution")]),
            "encode": lambda *_args: ([1], [2, 3]),
            "build_prompt": lambda _question: "prompt",
            "build_response": lambda _thought, _solution: "response",
        })

        with patch.dict(sys.modules, {"transformers": transformers_stub}):
            with contextlib.redirect_stdout(output):
                scope["cmd_inspect"](args)

        self.assertIn("Token lengths: median 3, p90 3, max 3", output.getvalue())


class BatchArgumentTests(unittest.TestCase):
    def test_plan_steps_rejects_nonpositive_batch_factors(self):
        plan_steps = load_functions("train.py", {"plan_steps"}, {"math": math})["plan_steps"]
        for batch_size, grad_accum in ((0, 1), (1, 0)):
            with self.subTest(batch_size=batch_size, grad_accum=grad_accum):
                with self.assertRaises(SystemExit):
                    plan_steps(8, batch_size, grad_accum, 1)

    def test_plan_batches_rejects_nonpositive_device_batch(self):
        plan_batches = load_functions(
            "grpo.py", {"plan_batches", "largest_divisor_up_to"}, {}
        )["plan_batches"]
        with self.assertRaises(SystemExit):
            plan_batches(num_generations=8, per_device_batch=0, prompts_per_step=2)


if __name__ == "__main__":
    unittest.main()
