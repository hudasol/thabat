import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_adoption_example_runs(capsys):
    runpy.run_path(str(ROOT / "examples" / "adopt_minimal.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "thr:" in out and "fp64:" in out
