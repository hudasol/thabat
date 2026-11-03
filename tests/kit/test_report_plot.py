import matplotlib
import pytest

from thabat_kit import plotstyle, report


def test_template_strict():
    assert report.fill_template("a {{x:.2f}} b {{y}}", {"x": 0.123456, "y": 7}) == "a 0.12 b 7"
    with pytest.raises(KeyError):
        report.fill_template("{{missing}}", {})


def test_is_current(tmp_path):
    t = tmp_path / "t.md"
    r = tmp_path / "r.md"
    t.write_text("n = {{n}}")
    report.render_file(t, {"n": 3}, r)
    assert report.is_current(t, {"n": 3}, r) and not report.is_current(t, {"n": 4}, r)


def test_plotstyle_writes_file(tmp_path):
    plotstyle.apply_style()
    fig, ax = matplotlib.pyplot.subplots()
    plotstyle.interval_plot(ax, ["a", "b"], [0.2, 0.5], [0.1, 0.3], [0.3, 0.7], ref=0.5, xlabel="rate")
    out = tmp_path / "f.png"
    plotstyle.save(fig, out)
    assert out.stat().st_size > 500
