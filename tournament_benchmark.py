"""
Root wrapper that delegates to training/tournament_benchmark.py.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))

from tournament_benchmark import main

if __name__ == "__main__":
    main()
