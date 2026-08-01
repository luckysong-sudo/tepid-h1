from __future__ import annotations

import importlib
import re
import unittest
from pathlib import Path


DOCUMENTED_MODULES = [
    "tepid_h1",
    "tepid_h1.agent",
    "tepid_h1.data",
    "tepid_h1.evaluation",
    "tepid_h1.integrations",
    "tepid_h1.modeling",
]


def _documented_exports(document: str, module_name: str) -> list[str]:
    match = re.search(
        rf"^## `{re.escape(module_name)}`\n(?P<body>.*?)(?=^## `|\Z)",
        document,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing API reference section for {module_name}")
    return re.findall(r"^- `([^`]+)` - .+$", match.group("body"), flags=re.MULTILINE)


class ApiReferenceDocsTests(unittest.TestCase):
    def test_api_reference_tracks_public_exports(self) -> None:
        document = Path("docs/API_REFERENCE.md").read_text(encoding="utf-8")

        for module_name in DOCUMENTED_MODULES:
            module = importlib.import_module(module_name)
            with self.subTest(module=module_name):
                self.assertEqual(_documented_exports(document, module_name), module.__all__)


if __name__ == "__main__":
    unittest.main()
