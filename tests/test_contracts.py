"""Import contract and house style rules, enforced by test."""
import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def test_kit_never_imports_study():
    for f in (ROOT / "src" / "thabat_kit").glob("*.py"):
        assert "thabat_study" not in imported_modules(f), f


def test_kit_does_not_import_torch_at_module_level():
    for f in (ROOT / "src" / "thabat_kit").glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                assert not any((n or "").split(".")[0] == "torch" for n in names), f


def tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [ROOT / p for p in out if p.endswith((".md", ".py", ".toml", ".json", ".ipynb", ".txt"))]


def test_house_style():
    em_dash = "—"
    banned = ["Cl" + "aude", "Anth" + "ropic", "Univer" + "sity of", "Stir" + "ling", "Co-Authored"]
    for f in tracked_text_files():
        if f.name == "test_contracts.py" or f.name == "LICENSE":
            continue
        text = f.read_text(errors="ignore")
        assert em_dash not in text, f"em dash in {f}"
        for word in banned:
            assert word not in text, f"{word!r} in {f}"
