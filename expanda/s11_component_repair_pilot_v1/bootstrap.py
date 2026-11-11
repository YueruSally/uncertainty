"""Reuse the immutable S11 evaluator; this version has separate code and outputs."""
from pathlib import Path
import sys
NEW = Path(__file__).resolve().parent
sys.path.insert(0, str(NEW.parent / 's11_targeted_v1'))
from common import *
import run as legacy_run
import learning


def execute_jobs(*args, **kwargs):
    # The legacy supervisor is reused, but its child entry point is this version.
    before = legacy_run.ROOT
    legacy_run.ROOT = NEW
    try:
        return legacy_run.execute_jobs(*args, **kwargs)
    finally:
        legacy_run.ROOT = before
