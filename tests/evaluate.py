import os
import sys
import pytest

# Ensure parent directory is in path so we can import test modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.test_config
import tests.test_core
import tests.test_qa_utils
import tests.test_utils


class RewardCollector:
    """Pytest plugin to count passed and failed tests programmatically."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.total += 1
            if report.passed:
                self.passed += 1
            else:
                self.failed += 1

def main():
    # Prioritize the agent's submission directory in python path
    submission_path = "/workspace/submission"
    sys.path.insert(0, submission_path)
    
    # Fallback to the current directory (helps when running verifier on the solution itself)
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(1, root_path)

    # Instantiate reward collector plugin
    collector = RewardCollector()

    # Run pytest programmatically on all test files in tests/
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pytest.main(["-v", "--tb=short", test_dir], plugins=[collector])

    # Calculate final reward (ratio of passed tests to total tests)
    total_tests = collector.total
    passed_tests = collector.passed
    reward = (passed_tests / total_tests) if total_tests > 0 else 0.0

    print(f"\n📊 Verifier Summary:")
    print(f"   Total Tests:  {total_tests}")
    print(f"   Passed Tests: {passed_tests}")
    print(f"   Failed Tests: {collector.failed}")
    print(f"   Final Reward: {reward:.4f}")

    # Write reward to the standard location expected by the Harbor framework
    reward_dir = "/logs/verifier"
    try:
        os.makedirs(reward_dir, exist_ok=True)
    except OSError:
        # Fallback to local directory if root /logs is read-only (e.g. running locally on macOS)
        reward_dir = os.path.join(root_path, "logs/verifier")
        os.makedirs(reward_dir, exist_ok=True)
        
    reward_path = os.path.join(reward_dir, "reward.txt")
    with open(reward_path, "w") as f:
        f.write(f"{reward:.4f}\n")

    print(f"✅ Reward written to {reward_path}")

if __name__ == "__main__":
    main()
