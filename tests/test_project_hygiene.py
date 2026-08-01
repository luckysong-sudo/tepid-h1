from __future__ import annotations

import re
import subprocess
import unittest
from fnmatch import fnmatch
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

import tepid_h1


ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    if match is None:
        raise AssertionError("pyproject.toml must declare a project version")
    return match.group(1)


class ProjectVersionTests(unittest.TestCase):
    def test_package_version_matches_project_metadata(self) -> None:
        self.assertEqual(tepid_h1.__version__, _pyproject_version())

    def test_cli_version_matches_project_metadata(self) -> None:
        from tepid_h1.cli import build_parser

        with patch("sys.stdout", new=StringIO()) as stdout:
            with self.assertRaises(SystemExit) as raised:
                build_parser().parse_args(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), f"tepid-h1 {_pyproject_version()}")


class RepositoryHygieneTests(unittest.TestCase):
    def test_local_artifacts_are_not_tracked(self) -> None:
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            pytest.skip(f"git tracked-file check is unavailable: {exc!r}")

        tracked = set(result.stdout.splitlines())
        forbidden_exact = {"$null"}
        forbidden_patterns = {"python-*-amd64.exe"}
        offenders = sorted(
            path
            for path in tracked
            if path in forbidden_exact
            or any(fnmatch(path, pattern) for pattern in forbidden_patterns)
        )
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
