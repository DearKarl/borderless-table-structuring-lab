"""Original PNG/JPEG entry for the frozen hybrid_v1 system."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid.hybrid_v1.predict import main
if __name__ == '__main__': main()
