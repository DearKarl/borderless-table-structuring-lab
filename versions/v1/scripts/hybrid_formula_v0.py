"""Standalone cached assembly entry; does not load models."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hybrid.formula_v0.cli import main

if __name__=='__main__':main()
