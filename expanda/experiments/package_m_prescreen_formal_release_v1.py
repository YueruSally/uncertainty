#!/usr/bin/env python3
"""Build a test-free, checksummed repository and formal-result release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


RELEASE_NAME = "CCP_M_Transformer_PPO_K8_Formal_v1"
RELEASE_SOURCE = Path("expanda/releases/ccp_m_transformer_ppo_k8_formal_v1")
REQUIRED_RESULTS = {
    "ccp_m_prescreen_final_oracle_v1_3inst_10seed_b400":
        "PIPELINE_COMPLETE",
    "ccp_m_prescreen_scale_instances_v1": "catalog.json",
    "ccp_m_prescreen_scale_collection_v2": "COLLECTION_COMPLETE",
    "ccp_m_prescreen_scale_curve_v2": "TRAINING_COMPLETE",
    "ccp_m_prescreen_scale_search_instances_v1": "catalog.json",
    "ccp_m_prescreen_scale_search_v1_3inst_10seed_b400": "PIPELINE_COMPLETE",
    "ccp_m_prescreen_ppo_training_v1": "TRAINING_COMPLETE",
    "ccp_m_prescreen_ppo_selection_v1_20valid": "PIPELINE_COMPLETE",
    "ccp_m_prescreen_locked_final_instances_v1": "catalog.json",
    "ccp_m_prescreen_formal_v1_10inst_10seed_b400_oos5000":
        "PIPELINE_COMPLETE",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded(relative, result_tree=False):
    parts = relative.parts
    name = relative.name
    if ("tests" in parts or name.startswith("test_")
            or name in ("tests.log", ".lock")
            or "__pycache__" in parts or ".pytest_cache" in parts
            or name.endswith((".pyc", ".pyo"))):
        return True
    if result_tree and ("logs" in parts or "interrupted_runs" in parts
                        or name.endswith(".log")):
        return True
    return False


def copy_tracked_repository(repo, target):
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z"])
    files = [Path(value.decode()) for value in raw.split(b"\0") if value]
    selected = [path for path in files if not excluded(path)]
    for index, relative in enumerate(selected, 1):
        source = repo/relative
        if not source.is_file():
            raise FileNotFoundError(f"tracked source is missing: {source}")
        destination = target/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if index % 500 == 0:
            print(f"[M-RELEASE] repository {index}/{len(selected)}", flush=True)
    return len(files), len(selected)


def copy_tree(source, target):
    count = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if excluded(relative, result_tree=True):
            continue
        destination = target/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination); count += 1
    return count


def inventory(root, include_manifest=False):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.name == "SHA256SUMS":
            continue
        if relative.name == "RELEASE_MANIFEST.json" and not include_manifest:
            continue
        rows.append({"path": str(relative), "bytes": path.stat().st_size,
                     "sha256": sha256(path)})
    return rows


def execute(a):
    repo = a.repo.resolve(); diagnostics = a.diagnostics.resolve()
    out = a.out.resolve(); archive = out.with_suffix(".tar.gz")
    checksum = archive.with_suffix(archive.suffix+".sha256")
    if out.exists() or archive.exists() or checksum.exists():
        raise ValueError("release output already exists; choose a new --out")
    if subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--git-dir"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL).returncode != 0:
        raise ValueError("--repo is not a Git worktree")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True)
    if status.strip():
        raise ValueError("repository must be clean before packaging")
    for name, marker in REQUIRED_RESULTS.items():
        if not (diagnostics/name/marker).is_file():
            raise FileNotFoundError(f"required result is incomplete: {name}/{marker}")
    if not a.normalization.is_file() or not (a.source_instances/"catalog.json").is_file():
        raise FileNotFoundError("normalization or frozen source instances missing")

    out.mkdir(parents=True); started = time.monotonic()
    repository = out/"repository"
    total_tracked, copied_tracked = copy_tracked_repository(repo, repository)
    results = out/"results"; result_counts = {}
    for index, name in enumerate(REQUIRED_RESULTS, 1):
        result_counts[name] = copy_tree(diagnostics/name, results/name)
        print(f"[M-RELEASE] result {index}/{len(REQUIRED_RESULTS)} "
              f"{name} files={result_counts[name]}", flush=True)
    external = out/"external_inputs"
    result_counts["source_instances"] = copy_tree(
        a.source_instances.resolve(), external/"source_instances")
    external.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.normalization.resolve(), external/"reward_normalization.json")
    release_source = repo/RELEASE_SOURCE
    for name in ("README_CN.md", "REPRODUCE_CN.md", "RESULT_SUMMARY.json",
                 "RESULT_TABLE.csv"):
        shutil.copy2(release_source/name, out/name)

    forbidden = [path for path in out.rglob("*")
                 if path.is_file() and excluded(path.relative_to(out))]
    if forbidden:
        raise RuntimeError(f"test/cache files entered release: {forbidden[:5]}")
    rows = inventory(out)
    manifest = {"release": RELEASE_NAME, "source_commit": commit,
        "created_unix_time": time.time(),
        "repository_tracked_files": total_tracked,
        "repository_files_after_test_exclusion": copied_tracked,
        "result_file_counts": result_counts,
        "excluded": ["tests/", "test_*.py", "tests.log",
                     "result pipeline/worker *.log", "__pycache__/",
                     ".pytest_cache/", "interrupted_runs/", ".git/"],
        "file_count": len(rows),
        "total_bytes_before_archive": sum(row["bytes"] for row in rows),
        "files": rows}
    (out/"RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    # SHA256SUMS covers every release file except itself, including the
    # manifest.  The manifest covers every payload file that existed before
    # the two control files were written.
    all_rows = inventory(out, include_manifest=True)
    (out/"SHA256SUMS").write_text("".join(
        f"{row['sha256']}  {row['path']}\n" for row in all_rows),
        encoding="utf-8")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(out, arcname=out.name)
    archive_sha = sha256(archive)
    checksum.write_text(f"{archive_sha}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"completed": True, "release_directory": str(out),
        "archive": str(archive), "archive_sha256": archive_sha,
        "archive_bytes": archive.stat().st_size,
        "file_count": len(all_rows),
        "tests_included": False,
        "elapsed_seconds": time.monotonic()-started}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--source-instances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
