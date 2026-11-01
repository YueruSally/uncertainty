#!/usr/bin/env python3
"""Build a compact downloadable source archive for the frozen M and I rules."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import subprocess
import tarfile


RELEASE_DIR = Path(__file__).resolve().parent
REPOSITORY = RELEASE_DIR.parents[2]
PACKAGE_NAME = "ccp_m_i_directed_components_v1"
OUTPUT = REPOSITORY / f"{PACKAGE_NAME}_source.tar.gz"
M_TAG = "ccp-m-directed-balanced-v1"

CURRENT_FILES = [
    "requirements.txt",
    "expanda/requirements.txt",
    "expanda/requirements_transformer_ppo.txt",
    "expanda/baseline_uncertainty.py",
    "expanda/multi_instance_logging.py",
    "expanda/run_learning_components_deterministic.py",
    "expanda/run_ccp_candidate_pool.py",
    "expanda/run_ev_ccp_oos_pilot.py",
    "expanda/run_formal_ev_vs_ccp_s30_30runs.py",
    "expanda/evaluate_m_late_replace_frequency.py",
    "expanda/evaluate_m_directed_balanced_oos.py",
    "expanda/run_m_directed_replace_screen_v1.sh",
    "expanda/run_m_late_replace_frequency_v1.sh",
    "expanda/run_m_directed_balanced_validation_v1.sh",
    "expanda/run_m_directed_balanced_oos_v1.sh",
    "expanda/evaluate_i_directed_initialization.py",
    "expanda/analyze_i_initial_quality_gate.py",
    "expanda/run_i_directed_initialization_screen_v1.sh",
    "expanda/run_i_balanced_strength_grid_v1.sh",
    "expanda/run_i_directed_balanced_validation_v1.sh",
    "expanda/run_i_directed_balanced_hv_gate_validation_v1.sh",
    "expanda/run_i_initial_quality_gate_v1.sh",
    "expanda/run_i_directed_balanced_oos_v1.sh",
    "expanda/tests/test_learning_components_v2_controls.py",
    "expanda/tests/test_m_late_replace_frequency.py",
    "expanda/tests/test_m_directed_balanced_oos.py",
    "expanda/tests/test_initialization_rule.py",
]

M_TAG_FILES = [
    "requirements.txt",
    "expanda/requirements.txt",
    "expanda/requirements_transformer_ppo.txt",
    "expanda/baseline_uncertainty.py",
    "expanda/multi_instance_logging.py",
    "expanda/run_learning_components_deterministic.py",
    "expanda/run_ccp_candidate_pool.py",
    "expanda/run_ev_ccp_oos_pilot.py",
    "expanda/run_formal_ev_vs_ccp_s30_30runs.py",
    "expanda/evaluate_m_late_replace_frequency.py",
    "expanda/evaluate_m_directed_balanced_oos.py",
    "expanda/run_m_directed_replace_screen_v1.sh",
    "expanda/run_m_late_replace_frequency_v1.sh",
    "expanda/run_m_directed_balanced_validation_v1.sh",
    "expanda/run_m_directed_balanced_oos_v1.sh",
    "expanda/tests/test_learning_components_v2_controls.py",
    "expanda/tests/test_m_late_replace_frequency.py",
    "expanda/tests/test_m_directed_balanced_oos.py",
]


def tracked_files(prefix: str):
    output = subprocess.check_output(
        ["git", "ls-files", prefix], cwd=REPOSITORY, text=True)
    return [line for line in output.splitlines() if line]


def add_bytes(archive, name, data, mode=0o644):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    archive.addfile(info, io.BytesIO(data))


def main():
    current = list(CURRENT_FILES)
    current.extend(tracked_files("expanda/transformer_ppo"))
    current.extend(tracked_files("expanda/data"))
    current.extend(tracked_files("expanda/releases/ccp_i_directed_balanced_v1"))
    current.extend(tracked_files("expanda/releases/ccp_m_i_directed_components_v1"))
    current = sorted(set(current))

    tagged = list(M_TAG_FILES)
    tag_listing = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", M_TAG],
        cwd=REPOSITORY, text=True).splitlines()
    tagged.extend(
        path for path in tag_listing
        if path.startswith("expanda/transformer_ppo/")
        or path.startswith("expanda/data/")
        or path.startswith("expanda/releases/ccp_m_directed_balanced_v1/"))
    tagged = sorted(set(tagged))

    with tarfile.open(OUTPUT, "w:gz") as archive:
        for relative in current:
            path = REPOSITORY / relative
            if not path.is_file():
                raise FileNotFoundError(relative)
            archive.add(
                path,
                arcname=f"{PACKAGE_NAME}/current_source/{relative}")

        for relative in tagged:
            data = subprocess.check_output(
                ["git", "show", f"{M_TAG}:{relative}"], cwd=REPOSITORY)
            mode = 0o755 if relative.endswith((".py", ".sh")) else 0o644
            add_bytes(
                archive,
                f"{PACKAGE_NAME}/frozen_m_tag/{relative}",
                data,
                mode=mode)

        for name in ("README_CN.md", "UPLOAD_TO_GITHUB.md"):
            data = (RELEASE_DIR / name).read_bytes()
            add_bytes(archive, f"{PACKAGE_NAME}/{name}", data)

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    checksum = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
    checksum.write_text(f"{digest}  {OUTPUT.name}\n", encoding="utf-8")
    print(OUTPUT)
    print(checksum)
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
