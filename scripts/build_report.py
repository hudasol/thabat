"""Run the thabat_kit tests with coverage and write results/build.json (read by the scorer)."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    xml, cov = ROOT / "results" / ".junit.xml", ROOT / "results" / ".cov.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/kit", "-q", "-p", "no:cacheprovider", "--cov=thabat_kit",
         f"--cov-report=json:{cov}", f"--junitxml={xml}"],
        cwd=ROOT, check=False, capture_output=True,
    )
    suite = ET.parse(xml).getroot()
    suite = suite if suite.tag == "testsuite" else suite[0]
    failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    percent = json.loads(cov.read_text())["totals"]["percent_covered"]
    out = {"tests_total.kit": int(suite.get("tests", 0)), "tests_failed.kit": failed,
           "coverage_percent.kit": round(percent, 1)}
    (ROOT / "results" / "build.json").write_text(json.dumps(out, indent=1))
    xml.unlink()
    cov.unlink()
    print(out)


if __name__ == "__main__":
    main()
