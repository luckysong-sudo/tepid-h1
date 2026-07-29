import ast
import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "integrations" / "huggingface-zero-gpu"


class ZeroGPUBundleTests(unittest.TestCase):
    def test_bundle_is_parseable_and_dependency_is_commit_pinned(self):
        ast.parse((BUNDLE / "app.py").read_text(encoding="utf-8"))
        requirements = (BUNDLE / "requirements.txt").read_text(encoding="utf-8")
        match = re.search(r"tepid-h1.+@([0-9a-f]{40})", requirements)
        self.assertIsNotNone(match)
        self.assertIn("sdk: gradio", (BUNDLE / "README.md").read_text(encoding="utf-8"))
        app_source = (BUNDLE / "app.py").read_text(encoding="utf-8")
        self.assertIn("/data/reports", app_source)
        self.assertIn('allowed_paths=["/data/reports"]', app_source)

    def test_bundled_corpus_matches_inventory_digest(self):
        corpus = (BUNDLE / "paired_corpus.jsonl").read_bytes()
        inventory = json.loads((BUNDLE / "data_inventory.json").read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(corpus).hexdigest(),
            inventory["sources"][0]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
