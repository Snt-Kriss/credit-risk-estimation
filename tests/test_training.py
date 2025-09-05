import subprocess
import sys
def test_training_runs():
    """Run classifier.py and ensure it executes without crashing"""
    result= subprocess.run(
        [sys.executable, "training/classifier.py"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Training failed: {result.stderr}"