#!/usr/bin/env python3
"""Run CCP100 with eligibility-conditioned mutation-location selection."""
import sys

from run_learning_ccp100 import main


if __name__ == "__main__":
    raise SystemExit(main(["--policy", "rule", *sys.argv[1:]]))
