"""Fill document templates strictly from a metrics dict, so documents cannot drift from results."""
from __future__ import annotations

import re
from pathlib import Path

PLACEHOLDER = re.compile(r"\{\{\s*([\w.]+)(?::([^}\s]+))?\s*\}\}")


def fill_template(template: str, metrics: dict) -> str:
    """Replace {{key}} or {{key:.2f}}. A missing key raises KeyError."""

    def sub(m: re.Match) -> str:
        key, spec = m.group(1), m.group(2)
        if key not in metrics:
            raise KeyError(f"template key not in metrics: {key}")
        v = metrics[key]
        return format(v, spec) if spec else str(v)

    return PLACEHOLDER.sub(sub, template)


def render_file(template_path, metrics: dict, out_path) -> str:
    text = fill_template(Path(template_path).read_text(), metrics)
    Path(out_path).write_text(text)
    return text


def is_current(template_path, metrics: dict, rendered_path) -> bool:
    """True when the rendered file equals a fresh rendering of the template."""
    return fill_template(Path(template_path).read_text(), metrics) == Path(rendered_path).read_text()
