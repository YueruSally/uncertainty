#!/usr/bin/env python3
"""Verify the Directed Balanced I core files and recorded result snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RELEASE_DIR = Path(__file__).resolve().parent
REPOSITORY = RELEASE_DIR.parents[2]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    manifest = read_json(RELEASE_DIR / "VERSION_MANIFEST.json")
    failures = []
    for relative, expected in manifest["core_files_sha256"].items():
        path = REPOSITORY / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(
                f"checksum mismatch: {relative}: {actual} != {expected}")

    results = read_json(RELEASE_DIR / "RESULTS.json")
    oos = results["independent_oos_heldout_70_pairs"]
    interpretation = results["interpretation"]
    if oos["mean_paired_oos_percent"] <= 10.0:
        failures.append("recorded held-out OOS mean does not exceed 10%")
    if oos["mean_95ci_low"] <= 0.0:
        failures.append("recorded held-out OOS confidence interval includes zero")
    if oos["wins"] + oos["ties"] + oos["losses"] != 70:
        failures.append("recorded held-out OOS counts do not total 70")
    if interpretation["strict_run_level_safety_pass"] is not False:
        failures.append("strict safety caveat was removed")
    if interpretation["ppo_has_learned_this_result"] is not False:
        failures.append("claim boundary was changed")
    if results["rejected_hv_gate"]["accepted_for_release"] is not False:
        failures.append("rejected HV gate was marked as accepted")

    if failures:
        raise SystemExit(
            "CCP-I RELEASE VERIFICATION FAILED\n" + "\n".join(failures))
    print("CCP-I Directed Balanced v1 release verification: OK")


if __name__ == "__main__":
    main()
