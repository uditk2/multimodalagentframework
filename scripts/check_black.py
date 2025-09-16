#!/usr/bin/env python3
"""
Black formatting check script for CI/CD pipeline.
This script checks if code is properly formatted with black and provides clear feedback.
"""

import subprocess
import sys
from pathlib import Path


def run_black_check():
    """Run black --check on the codebase and provide clear feedback."""

    # Define the paths to check
    paths_to_check = ["multimodal_agent_framework/", "tests/", "scripts/"]

    # Filter to only existing paths
    existing_paths = [path for path in paths_to_check if Path(path).exists()]

    if not existing_paths:
        print("❌ No Python code directories found to check")
        return 1

    print("🔍 Checking code formatting with black...")
    print(f"Checking paths: {', '.join(existing_paths)}")

    try:
        # Run black --check on the existing paths
        result = subprocess.run(
            ["black", "--check", "--diff"] + existing_paths,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ All Python code is properly formatted with black!")
            return 0
        else:
            print("❌ Code formatting issues found!")
            print("\nThe following files need formatting:")
            print(result.stdout)

            if result.stderr:
                print("\nErrors:")
                print(result.stderr)

            print("\n💡 To fix formatting issues, run:")
            print(f"   black {' '.join(existing_paths)}")
            return 1

    except FileNotFoundError:
        print("❌ Black is not installed. Please install it with:")
        print("   pip install black")
        return 1
    except Exception as e:
        print(f"❌ Error running black: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_black_check())
