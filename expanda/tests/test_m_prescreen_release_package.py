import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE / "experiments" / "package_m_prescreen_formal_release_v1.py"
SPEC = importlib.util.spec_from_file_location("m_release_package", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class FormalReleasePackageTests(unittest.TestCase):
    def test_test_and_cache_paths_are_excluded_without_excluding_final_test(self):
        self.assertTrue(release.excluded(Path("expanda/tests/test_x.py")))
        self.assertTrue(release.excluded(Path("expanda/test_x.py")))
        self.assertTrue(release.excluded(Path("tests.log")))
        self.assertTrue(release.excluded(Path("x/__pycache__/a.pyc")))
        self.assertFalse(release.excluded(
            Path("results/final-test-S0-001/RUN_COMPLETE.json")))

    def test_result_copy_omits_logs_but_preserves_checkpoints_and_raw_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "source"; target = root / "target"
            (source / "logs").mkdir(parents=True)
            (source / "raw" / "final-test-S0-001").mkdir(parents=True)
            (source / "model.pt").write_bytes(b"checkpoint")
            (source / "QUICK_RESULT.json").write_text("{}\n")
            (source / "pipeline.nohup.log").write_text("noise\n")
            (source / "logs" / "worker.log").write_text("noise\n")
            (source / "raw" / "final-test-S0-001" / "row.json").write_text(
                "{}\n")
            count = release.copy_tree(source, target)
            self.assertEqual(count, 3)
            self.assertTrue((target / "model.pt").is_file())
            self.assertTrue((target / "QUICK_RESULT.json").is_file())
            self.assertTrue((target / "raw/final-test-S0-001/row.json").is_file())
            self.assertFalse((target / "pipeline.nohup.log").exists())

    def test_checksum_inventory_includes_manifest_only_when_requested(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "payload.txt").write_text("payload\n")
            (root / "RELEASE_MANIFEST.json").write_text("{}\n")
            (root / "SHA256SUMS").write_text("placeholder\n")
            payload = release.inventory(root)
            checksum = release.inventory(root, include_manifest=True)
            self.assertEqual([row["path"] for row in payload], ["payload.txt"])
            self.assertEqual([row["path"] for row in checksum],
                             ["RELEASE_MANIFEST.json", "payload.txt"])

    def test_end_to_end_build_is_test_free_and_checksummed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); repo = root / "repo"; diagnostics = root / "diag"
            repo.mkdir(); subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"],
                           check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email",
                            "test@example.invalid"], check=True)
            (repo / "app.py").write_text("print('ok')\n")
            (repo / "tests").mkdir(); (repo / "tests/test_app.py").write_text("pass\n")
            release_source = repo / release.RELEASE_SOURCE
            release_source.mkdir(parents=True)
            for name in ("README_CN.md", "REPRODUCE_CN.md",
                         "RESULT_SUMMARY.json", "RESULT_TABLE.csv"):
                (release_source / name).write_text("{}\n")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"],
                           check=True)

            for name, marker in release.REQUIRED_RESULTS.items():
                result = diagnostics / name; result.mkdir(parents=True)
                (result / marker).write_text("complete\n")
                (result / "model.pt").write_bytes(b"model")
                (result / "tests.log").write_text("excluded\n")
            instances = root / "instances"; instances.mkdir()
            (instances / "catalog.json").write_text("{}\n")
            (instances / "source.json").write_text("{}\n")
            normalization = root / "normalization.json"
            normalization.write_text("{}\n")
            out = root / "release"
            release.execute(SimpleNamespace(repo=repo, diagnostics=diagnostics,
                normalization=normalization, source_instances=instances, out=out))
            self.assertTrue((out / "RELEASE_MANIFEST.json").is_file())
            self.assertTrue((out / "SHA256SUMS").is_file())
            self.assertTrue(out.with_suffix(".tar.gz").is_file())
            self.assertFalse((out / "repository/tests").exists())
            self.assertFalse(list(out.rglob("tests.log")))
            checksums = (out / "SHA256SUMS").read_text()
            self.assertIn("RELEASE_MANIFEST.json", checksums)


if __name__ == "__main__":
    unittest.main()
