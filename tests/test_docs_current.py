"""Committed result documents must equal a fresh rendering from their templates and results/summary.json."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_render():
    spec = importlib.util.spec_from_file_location("render_results", ROOT / "scripts" / "render_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["RESULTS", "PAPER"])
def test_document_matches_source_files(name):
    tpl = ROOT / "docs" / f"{name}.template.md"
    if not tpl.exists():
        pytest.skip(f"{name} template not written yet")
    from thabat_kit import report

    ctx = load_render().blocks(ROOT)
    assert report.is_current(tpl, ctx, ROOT / "docs" / f"{name}.md"), f"{name}.md differs from its sources; run scripts/render_results.py"
