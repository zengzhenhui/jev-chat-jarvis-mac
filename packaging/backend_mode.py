"""Resolve installation mode using runtime configuration, without ML imports."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import userconfig

if __name__ == "__main__":
    print("local" if userconfig.get("JUDGE_BACKEND").strip().lower() == "local" else "api")
