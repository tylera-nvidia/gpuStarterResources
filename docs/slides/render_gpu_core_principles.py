#!/usr/bin/env python3
"""Render the GPU Core Principles HTML deck to PDF and PNG previews."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "gpu-core-principles.html"
PDF = HERE / "gpu-core-principles.pdf"
PREVIEW_DIR = HERE / "previews"


def find_chrome() -> Path:
    override = os.environ.get("CHROME_BIN")
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        raise SystemExit(f"CHROME_BIN does not point to a file: {candidate}")

    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return Path(found)

    cached_shell = sorted(
        Path("/scratch/tylera/cache/ms-playwright").glob(
            "chromium_headless_shell-*/chrome-linux/headless_shell"
        )
    )
    if cached_shell:
        return cached_shell[-1]

    cached = sorted(Path("/scratch/tylera/cache/ms-playwright").glob("chromium-*/chrome-linux/chrome"))
    if cached:
        return cached[-1]

    raise SystemExit("Chromium/Chrome not found. Set CHROME_BIN to a compatible executable.")


def run_chrome(chrome: Path, profile: Path, *args: str) -> None:
    command = [
        str(chrome),
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--user-data-dir={profile}",
        *args,
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def validate_pdf() -> None:
    payload = PDF.read_bytes()
    if not payload.startswith(b"%PDF-"):
        raise SystemExit(f"Output is not a PDF: {PDF}")

    page_count = len(re.findall(rb"/Type\s*/Page\b", payload))
    if page_count != 3:
        raise SystemExit(f"Expected 3 PDF pages, found {page_count}")

    media_boxes = set(re.findall(rb"/MediaBox\s*\[([^]]+)\]", payload))
    if media_boxes != {b"0 0 960 540"}:
        raise SystemExit(f"Expected 16:9 MediaBox 960x540, found {sorted(media_boxes)!r}")


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"Missing source deck: {SOURCE}")

    chrome = find_chrome()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gpu-slides-chrome-") as profile_name:
        profile = Path(profile_name)
        run_chrome(
            chrome,
            profile,
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={PDF}",
            SOURCE.as_uri(),
        )
        PDF.chmod(0o644)
        validate_pdf()

        for slide in range(1, 4):
            preview = PREVIEW_DIR / f"slide-{slide}.png"
            run_chrome(
                chrome,
                profile,
                "--window-size=1280,720",
                "--virtual-time-budget=500",
                f"--screenshot={preview}",
                f"{SOURCE.as_uri()}?slide={slide}",
            )
            preview.chmod(0o644)

    print(f"Wrote {PDF} (3 pages, 16:9)")
    print(f"Wrote previews to {PREVIEW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
