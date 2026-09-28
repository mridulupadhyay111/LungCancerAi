"""
run_app.py

Convenience Launcher script for the LungCancerAI Web Application.
Runs Streamlit server on http://localhost:8501
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    app_path = os.path.join("deployment", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    print("=" * 70)
    print("Launching LungCancerAI Diagnostic Web Application...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    subprocess.run(cmd)
