"""
drawer
------
Thin wrapper over the Mermaid CLI (``mmdc``) used to render the
architecture diagrams shipped with the thesis. Renders one or all
diagrams, returns the output paths and exposes helpers to fetch the
Mermaid source for embedding in the visualisation notebook.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from sources.config.graph import (
    GRAPH_DESCRIPTIONS as DESCRIPTIONS,
    GRAPH_DIAGRAMS as DIAGRAMS,
    GRAPH_TITLES as TITLES,
)


def _find_mmdc() -> str:
    """Locate the Mermaid CLI executable on the system PATH.

    Returns:
        str: Absolute path to the ``mmdc`` binary.

    Raises:
        FileNotFoundError: If ``mmdc`` is not found on PATH.
    """
    mmdc = shutil.which("mmdc")
    if mmdc is None:
        raise FileNotFoundError(
            "Mermaid CLI (mmdc) not found on PATH. "
            "Install via: npm install -g @mermaid-js/mermaid-cli"
        )
    return mmdc


def render_one(
    name: str,
    output_dir: str | Path = "output/diagrams",
    *,
    fmt: str = "svg",
    theme: str = "default",
    background: str = "transparent",
    width: int = 1200,
) -> Path:
    """Render one named diagram to a file and return the output path.

    Args:
        name (str): Diagram key from ``GRAPH_DIAGRAMS`` (e.g. ``"deterministic"``).
        output_dir (str | Path): Directory where the rendered file is written.
        fmt (str): Output format passed to mmdc (e.g. ``"svg"``, ``"png"``).
        theme (str): Mermaid theme name (e.g. ``"default"``, ``"dark"``).
        background (str): Background colour string (e.g. ``"transparent"``).
        width (int): Diagram width in pixels.

    Returns:
        Path: Path to the rendered output file.

    Raises:
        ValueError: If ``name`` is not a key in ``GRAPH_DIAGRAMS``.
        RuntimeError: If the mmdc process exits with a non-zero return code.
    """
    if name not in DIAGRAMS:
        raise ValueError(f"Unknown diagram '{name}'. Choose from: {list(DIAGRAMS)}")

    mmdc = _find_mmdc()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / f"{name}.{fmt}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mmd", delete=False) as tmp:
        tmp.write(DIAGRAMS[name])
        tmp_path = tmp.name

    try:
        cmd = [
            mmdc,
            "-i",
            tmp_path,
            "-o",
            str(dest),
            "-t",
            theme,
            "-b",
            background,
            "-w",
            str(width),
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"mmdc failed (code {result.returncode}):\n{result.stderr}"
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return dest


def render_all(
    output_dir: str | Path = "output/diagrams",
    *,
    fmt: str = "svg",
    theme: str = "default",
    background: str = "transparent",
    width: int = 1200,
) -> dict[str, Path]:
    """Render all diagrams to files and return their output paths.

    Args:
        output_dir (str | Path): Directory where all rendered files are written.
        fmt (str): Output format passed to mmdc (e.g. ``"svg"``, ``"png"``).
        theme (str): Mermaid theme name (e.g. ``"default"``, ``"dark"``).
        background (str): Background colour string (e.g. ``"transparent"``).
        width (int): Diagram width in pixels.

    Returns:
        dict[str, Path]: Mapping from diagram name to its rendered output path.
    """
    return {
        name: render_one(
            name,
            output_dir,
            fmt=fmt,
            theme=theme,
            background=background,
            width=width,
        )
        for name in DIAGRAMS
    }


def get_mermaid_source(name: str) -> str:
    """Return the Mermaid diagram source string for one diagram.

    Args:
        name (str): Diagram key from ``GRAPH_DIAGRAMS``.

    Returns:
        str: Raw Mermaid source text for the requested diagram.

    Raises:
        ValueError: If ``name`` is not a key in ``GRAPH_DIAGRAMS``.
    """
    if name not in DIAGRAMS:
        raise ValueError(f"Unknown diagram '{name}'. Choose from: {list(DIAGRAMS)}")
    return DIAGRAMS[name]
