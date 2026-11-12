"""Use the frozen full-budget search implementation without editing its files."""
from pathlib import Path
import importlib.util
import sys

FAST=Path(__file__).resolve().parent
FULL=FAST.parent/'joint_icm_multiscale_terminal_v1'
sys.path.insert(0,str(FULL))
from bridge import *
from prepare import verify_shared
from policy import TerminalPolicy
spec=importlib.util.spec_from_file_location('full_search_for_fastcheck',FULL/'run.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)


def runtime_files_fast():
    paths=list(FAST.glob('*.py'))+[FAST/'protocol.json',FAST.parent/'audit_s0_joint_v1/checkpoints.py']
    return {**files(),**{str(p):digest(p) for p in paths}}


def jobs(root,plan,specs,workers,stage):
    old=full.HERE2;full.HERE2=FAST
    try:return full.execute_jobs(root,plan,specs,workers,stage)
    finally:full.HERE2=old
