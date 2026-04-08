#!/usr/bin/env python3
"""
Sanity check: greedy (-obj 6) vs beam with width 1 (-obj 7 -beam 1).
Expect identical .tree exports (beam width <= 1 delegates to greedy in buildBeamExact).

Paths are resolved relative to this script so it works from any cwd.
"""
from __future__ import annotations

import errno
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _is_elf(path: Path) -> bool:
    """True if file is a Linux/macOS ELF (cannot run natively on Windows)."""
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def find_solver(born_again_dp: Path) -> Optional[Path]:
    """
    Prefer bornAgain.exe on Windows. Skip a bare 'bornAgain' file if it is ELF
    (e.g. WSL build checked in by mistake) — that triggers WinError 193 on native Windows.
    """
    for name in ("bornAgain.exe", "bornAgain"):
        p = born_again_dp / name
        if not p.is_file():
            continue
        if sys.platform == "win32" and name == "bornAgain" and _is_elf(p):
            continue
        return p
    return None


def _solver_hint(born_again_dp: Path) -> None:
    bare = born_again_dp / "bornAgain"
    if sys.platform == "win32" and bare.is_file() and _is_elf(bare):
        print(
            "NOTE: 'bornAgain' in this folder is a Linux ELF binary and cannot run on native Windows.",
            file=sys.stderr,
        )
        print(
            "Build a Windows binary with MinGW:  cd src/born_again_dp  &&  mingw32-make",
            file=sys.stderr,
        )
        print(
            "Or run this sanity script from WSL using the Linux-bornAgain.",
            file=sys.stderr,
        )


def read_text_normalized(path: Path) -> str:
    raw = path.read_bytes()
    # UTF-8 with optional BOM; normalize newlines for comparison
    text = raw.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def main() -> int:
    here = Path(__file__).resolve().parent
    born_again_dp = here.parent
    rf = born_again_dp.parent / "resources" / "forests" / "Seeds" / "Seeds.RF1.txt"
    out_dir = here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ba = find_solver(born_again_dp)
    if ba is None:
        print(
            "ERROR: No runnable solver under",
            born_again_dp,
            "(need bornAgain.exe on Windows, or a non-ELF bornAgain from MinGW).",
            file=sys.stderr,
        )
        _solver_hint(born_again_dp)
        print("Build:  cd src/born_again_dp  &&  mingw32-make   (or make on Linux/WSL)", file=sys.stderr)
        return 1

    if not rf.is_file():
        print("ERROR: Missing forest file:", rf, file=sys.stderr)
        return 1

    base_g = out_dir / "greedy"
    base_b = out_dir / "beam_w1"
    tree_g = Path(str(base_g) + ".tree")
    tree_b = Path(str(base_b) + ".tree")

    def run(args: list[str]) -> None:
        try:
            r = subprocess.run(
                [str(ba), str(rf), str(args[0])] + args[1:],
                cwd=str(born_again_dp),
            )
        except OSError as e:
            if getattr(e, "winerror", None) == 193 or e.errno == errno.ENOEXEC:
                print("ERROR: This binary is not a valid Windows executable.", file=sys.stderr)
                _solver_hint(born_again_dp)
            else:
                print(f"ERROR: Could not run solver: {e}", file=sys.stderr)
            raise SystemExit(1)
        if r.returncode != 0:
            raise SystemExit(r.returncode)

    print("Running greedy (objective 6)...")
    run([str(base_g), "-obj", "6", "-trees", "1"])
    print("Running beam width 1 (objective 7, -beam 1)...")
    run([str(base_b), "-obj", "7", "-beam", "1", "-trees", "1"])

    if not tree_g.is_file() or not tree_b.is_file():
        print("ERROR: Missing output .tree file(s).", file=sys.stderr)
        return 1

    g = read_text_normalized(tree_g)
    b = read_text_normalized(tree_b)
    result_path = out_dir / "sanity_result.txt"
    if g == b:
        msg = "PASS: greedy.tree and beam_w1.tree are identical."
        result_path.write_text(msg + "\n", encoding="utf-8")
        print(msg)
        print("Wrote", result_path)
        return 0

    msg = "FAIL: tree files differ (see diff or compare manually)."
    result_path.write_text(msg + "\n", encoding="utf-8")
    print(msg, file=sys.stderr)
    print("Wrote", result_path)
    # Small hint: timing lines in .out may differ; .tree should match.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
