"""Расположение директорий лабораторной (не зависит от текущей рабочей директории)."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PACKAGE_DIR.parent
LAB_ROOT = SOURCE_DIR.parent
RESULTS_DIR = LAB_ROOT / "results"
TESTS_DIR = SOURCE_DIR / "tests"
