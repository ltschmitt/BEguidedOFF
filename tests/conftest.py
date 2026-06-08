# conftest for pytest
import sys
from pathlib import Path

# Ensure beguidedoff is importable from tests directory
sys.path.insert(0, str(Path(__file__).parent))
