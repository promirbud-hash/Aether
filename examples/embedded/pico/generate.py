from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.api import emit_c99, emit_c99_header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()

    source_path = Path(__file__).with_name("led_logic.hl")
    source = source_path.read_text(encoding="utf-8")
    output_directory = arguments.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "led_logic.c").write_text(
        emit_c99(source),
        encoding="utf-8",
        newline="\n",
    )
    (output_directory / "led_logic.h").write_text(
        emit_c99_header(source),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
