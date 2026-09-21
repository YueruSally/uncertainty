#!/usr/bin/env python3
"""Run CCP100 with an epsilon-greedy learned mutation-location policy."""
import sys

from run_learning_ccp100 import main


if __name__ == "__main__":
    raise SystemExit(main(["--policy", "learning", *sys.argv[1:]]))
