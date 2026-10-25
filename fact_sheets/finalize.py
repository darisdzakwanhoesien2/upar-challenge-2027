#!/usr/bin/env python3
"""Refuse to finish while a red TODO is left, then compile main.tex and zip the track folder for upar.challenge@gmail.com.

    python fact_sheets/finalize.py OUT_DIR/track1_attribute_recognition [--allow-todo]

Compiles with tectonic (or pdflatex + bibtex if tectonic is missing) and writes OUT_DIR/UPAR_FactSheet_<folder>.zip with the PDF, .tex, .bib and
the class files, which is what the organizers ask for.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--allow-todo", action="store_true", help="build a draft even though TODO fields remain")
    a = ap.parse_args()
    tex = (a.folder / "main.tex").read_text()
    todos = tex.count("[TODO") + tex.count("{TODO}")
    if todos and not a.allow_todo:
        print(f"{todos} TODO field(s) are still in {a.folder / 'main.tex'}; fill them in (search for TODO) or pass --allow-todo for a draft.")
        return 1
    if shutil.which("tectonic"):
        r = subprocess.run(["tectonic", "-X", "compile", "main.tex"], cwd=a.folder, capture_output=True, text=True)
    else:
        r = subprocess.run("pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex",
                           shell=True, cwd=a.folder, capture_output=True, text=True)
    if r.returncode != 0 or not (a.folder / "main.pdf").exists():
        print(r.stdout[-2000:], r.stderr[-2000:])
        return 1
    suffix = "_DRAFT" if todos else ""
    out = a.folder.parent / f"UPAR_FactSheet_{a.folder.name}{suffix}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("main.pdf", "main.tex", "references.bib", "ieeeconf.cls", "IEEEtran.cls", "IEEEtran.bst"):
            if (a.folder / f).exists():
                z.write(a.folder / f, f)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB), {todos} TODO field(s) left")
    return 0


if __name__ == "__main__":
    sys.exit(main())
